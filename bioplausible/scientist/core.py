"""
AutoScientist: The Autonomous Discovery Agent.

This module implements the core logic for the continuous experiment runner.
It manages the experiment lifecycle:
1. State Analysis: What have we learned so far?
2. Strategy: What should we do next? (Smoke -> Shallow -> Standard -> Deep)
3. Execution: Run the experiment.
4. Learning: Update the knowledge base.
"""

import json
import logging
import random
import signal
import sys
import time
from typing import Optional

import optuna  # noqa: F401

from bioplausible.hyperopt import (
    PatientLevel,
    create_constrained_optuna_config,
    get_evaluation_config,
)
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible.scientist.decisions import DecisionLogger
from bioplausible.scientist.resources import ResourceMonitor
from bioplausible.scientist.robustness import run_robustness_check
from bioplausible.scientist.state import ExperimentState
from bioplausible.scientist.strategy import ScientistStrategy
from bioplausible.scientist.task import ExperimentTask

# Re-export for backward compatibility
__all__ = [
    "AutoScientist",
    "ExperimentState",
    "ScientistStrategy",
    "ResourceMonitor",
    "ExperimentTask",
]

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scientist.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AutoScientist")

DB_PATH = "bioplausible.db"


class AutoScientist:
    """
    The main loop.
    """

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, db_path: str = DB_PATH, task_filter: Optional[str] = None):
        self.db_path = db_path
        self.state = ExperimentState(db_path)
        self.decision_logger = DecisionLogger(db_path)
        self.strategy = ScientistStrategy(self.state, self.decision_logger, task_filter=task_filter)
        self.resources = ResourceMonitor()
        self.running = True
        self.consecutive_failures = 0

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info("Interrupt received. Finishing current trial...")
        self.running = False

    def run(self):
        logger.info("AutoScientist initialized. Starting continuous discovery...")

        try:
            while self.running:
                # 0. Resource Check
                if self.resources.should_pause():
                    time.sleep(60)
                    continue

                # Check failures
                if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.critical(
                        f"Too many consecutive failures ({self.consecutive_failures}). Sleeping for 5 minutes."
                    )
                    time.sleep(300)
                    self.consecutive_failures = 0

                # 1. Plan
                task = self.strategy.plan_next()

                if not task:
                    logger.info("No viable experiments found. Sleeping 60s...")
                    time.sleep(60)
                    continue

                is_fixed = task.fixed_config is not None
                type_str = "EXPLORATION"
                if task.tier == PatientLevel.CROSS_VAL:
                    type_str = f"CROSS_VAL (Fold {task.fold_index})"
                elif is_fixed:
                    type_str = "VERIFICATION"
                elif task.is_robustness_check:
                    type_str = "ROBUSTNESS"
                elif task.is_ablation:
                    type_str = f"ABLATION ({task.ablation_param})"
                elif task.is_transfer:
                    type_str = f"TRANSFER (From #{task.transfer_from_trial})"
                elif task.is_continual:
                    type_str = f"CONTINUAL (Step {task.continual_step})"
                elif task.fixed_config and "data_fraction" in task.fixed_config:
                    type_str = f"LOW_DATA ({task.fixed_config['data_fraction']:.0%})"

                logger.info(
                    f"Starting {type_str}: {task.model_name} | {task.task_name} | "
                    f"{task.tier.name} (Priority: {task.priority:.1f})"
                )

                # 2. Prepare Config
                study = None
                trial = None

                try:
                    # Load Optuna Study
                    study = self.state.get_optuna_study(task.study_name)

                    config = {}
                    job_id = None

                    if is_fixed:
                        config = task.fixed_config
                        # Ensure fold is set for CV
                        if task.fold_index is not None:
                            config["fold"] = task.fold_index

                        # Set job_id for fixed tasks to avoid #N/A logging
                        if task.tier == PatientLevel.CROSS_VAL:
                            job_id = f"CV-{task.verification_of_trial_id}-F{task.fold_index}"
                        elif task.verification_of_trial_id:
                            job_id = f"Ver-{task.verification_of_trial_id}"
                        elif task.is_transfer:
                            job_id = f"Transfer-{task.transfer_from_trial}"
                        elif task.is_continual:
                            job_id = f"CL-{task.continual_step}"
                        elif "data_fraction" in config:
                            job_id = f"LowData-{config['data_fraction']}"
                        else:
                            job_id = f"Fixed-{task.study_name}"

                    else:
                        # Warm-Start Logic
                        best_trial = None
                        if random.random() < 0.2:  # 20% chance to warm start
                            # Find best trial for this model/task
                            try:
                                # We need to query DB manually or use study if it has history
                                # Simple way: if study has trials, pick best
                                if len(study.trials) > 0:
                                    best_trial = study.best_trial
                                    if best_trial:
                                        logger.info(
                                            f"  > Warm-starting from Trial #{best_trial.number} (Acc: {best_trial.value:.2%})")
                                        # Enqueue with slight mutation? Optuna enqueue is exact.
                                        # To mutate, we'd need to manually adjust params and enqueue.
                                        # For now, just enqueue best to reinforce known good regions for TPE
                                        study.enqueue_trial(best_trial.params)
                            except Exception as e:
                                logger.warning(f"Warm start failed: {e}")

                        trial = study.ask()
                        # Pass dynamic constraints (intelligence)
                        constraints = {}
                        if task.constraints:
                            constraints.update(task.constraints)
                            logger.info(
                                f"  > Applying intelligent constraints: {constraints}")

                        config = create_constrained_optuna_config(
                            trial,
                            task.model_name,
                            custom_constraints=constraints,
                            task_name=task.task_name
                        )
                        # Fix Trial #N/A bug (Phase 1.1)
                        if trial.number is not None:
                            job_id = trial.number
                        elif hasattr(trial, "_trial_id"):
                            job_id = trial._trial_id
                        else:
                            job_id = "Unknown"

                        # Log metadata for reports
                        trial.set_user_attr("model_name", task.model_name)
                        trial.set_user_attr("task_name", task.task_name)
                        trial.set_user_attr("tier", task.tier.value)

                    # Inject Tier Config
                    tier_config = get_evaluation_config(task.tier)
                    config["epochs"] = tier_config.epochs
                    config["batch_size"] = tier_config.batch_size

                    # Early Stopping Injection (Phase 2.2)
                    config["early_stopping_patience"] = 3

                    # Metadata
                    config["tier"] = task.tier.value
                    config["task"] = task.task_name
                    config["model"] = task.model_name
                    if is_fixed:
                        config["is_verification"] = True
                        config["verified_trial_id"] = task.verification_of_trial_id

                    if task.is_robustness_check:
                        config["is_robustness_check"] = True

                    if task.is_ablation:
                        config["is_ablation"] = True
                        config["ablation_param"] = task.ablation_param
                        # Ablations are scientifically interesting, so save artifacts
                        config["save_artifacts"] = True

                    if task.is_transfer:
                        config["is_transfer"] = True
                        config["transfer_from"] = task.transfer_from_trial
                        # Also save artifacts for transfer results
                        config["save_artifacts"] = True

                    if task.is_continual:
                        config["is_continual"] = True
                        config["continual_step"] = task.continual_step
                        config["save_artifacts"] = True
                        if task.transfer_from_trial:
                            config["transfer_from"] = task.transfer_from_trial

                    # 3. Execute
                    # Identify interesting params for logging
                    ignore_keys = {
                        "epochs",
                        "batch_size",
                        "tier",
                        "task",
                        "model",
                        "job_id",
                        "save_artifacts",
                        "fold",
                        "is_verification",
                        "verified_trial_id",
                        "is_robustness_check",
                        "is_ablation",
                        "ablation_param",
                        "is_transfer",
                        "transfer_from",
                        "is_continual",
                        "continual_step",
                    }
                    interesting_params = {
                        k: v for k, v in config.items() if k not in ignore_keys
                    }

                    logger.info(
                        f"  > Trial #{job_id if job_id is not None else 'N/A'}: "
                        f"Epochs={config.get('epochs')}, Batch={config.get('batch_size')}. "
                        f"Params: {interesting_params}"
                    )

                    if task.is_robustness_check:
                        # Run Robustness Suite
                        logger.info("  > Running Robustness Suite...")

                        # Locate artifact if verified_trial_id is present
                        weights_path = None
                        if task.verification_of_trial_id:
                            # Try to find artifacts
                            # Ideally we would query archiver or storage, but here we can try a pattern
                            # or just train from scratch as fallback (RobustnessEvaluator handles this)
                            pass

                        score = run_robustness_check(
                            task.model_name, task.task_name, config, weights_path=weights_path
                        )
                        # We return a dummy metrics dict to store in DB
                        metrics = {
                            "accuracy": score,  # Store robustness score as accuracy for now
                            "loss": 0.0,
                            "robustness_score": score,
                            "time": 0.0,
                            "param_count": 0.0,
                        }
                    else:
                        quick = task.tier == PatientLevel.SMOKE

                        if trial:
                            trial.set_user_attr("config", json.dumps(config))

                        config["job_id"] = job_id

                        metrics = run_single_trial_task(
                            task=task.task_name,
                            model_name=task.model_name,
                            config=config,
                            storage_path=DB_PATH,
                            quick_mode=quick,
                        )

                    # 4. Report
                    if metrics:
                        acc = metrics.get("accuracy", 0.0)
                        loss = metrics.get("loss", float("inf"))
                        logger.info(f"  > Result: Accuracy={acc:.2%}, Loss={loss:.4f}")

                        if trial:
                            study.tell(trial, acc)

                        self.consecutive_failures = 0  # Success!
                    else:
                        logger.warning("  > Trial failed.")
                        if trial:
                            study.tell(trial, state=optuna.trial.TrialState.FAIL)

                        self.consecutive_failures += 1

                except Exception as e:
                    logger.error(f"Error executing trial: {e}", exc_info=True)
                    self.consecutive_failures += 1
                    time.sleep(5)

                time.sleep(1)

                # Cleanup Memory aggressively
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            logger.info("AutoScientist shutting down. Cleaning up...")
            self.state.close()
            logger.info("Shutdown complete.")

    def generate_reports(self, output_dir: str = "reports"):
        """
        Generates comprehensive Scientist++ reports with ML analysis, visualizations,
        statistical tests, and high-level synthesis insights.
        """
        logger.info("Generating Scientist++ Reports...")

        from pathlib import Path
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = Path(output_dir) / f"run_{timestamp}"
        report_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate comprehensive report using Modular ReportComposer (Phase 4)
        logger.info("Generating modular analysis report...")
        try:
            from bioplausible.scientist.report.composer import ReportComposer
            composer = ReportComposer(self.db_path, str(report_path))
            composer.generate_report()
            composer.close()
            logger.info(
                "✓ Modular report generated (01_summary.md, 03_leaderboards.md, FULL_REPORT.md)")
        except Exception as e:
            logger.error(f"Failed to generate core report: {e}", exc_info=True)
            logger.error(f"Failed to generate comprehensive report: {e}", exc_info=True)

        # 2. Generate high-level synthesis insights (additional perspective)
        logger.info("Generating research synthesis...")
        try:
            from bioplausible.scientist.synthesizer import ResearchSynthesizer
            synthesizer = ResearchSynthesizer(self.db_path)
            synthesis_result = synthesizer.synthesize_full_report()

            # Create synthesis subdirectory
            synthesis_path = report_path / "synthesis"
            synthesis_path.mkdir(exist_ok=True)

            # Save Synthesis JSON
            with open(synthesis_path / "research_synthesis.json", "w") as f:
                json.dump(synthesis_result, f, indent=2)

            # Generate Synthesis Narrative
            with open(synthesis_path / "SYNTHESIS.md", "w") as f:
                f.write(f"# Research Synthesis\n")
                f.write(f"Generated: {timestamp}\n\n")

                # Cross-Algorithm Rankings
                f.write("## 🏆 Cross-Algorithm Performance Rankings\n\n")
                insights = synthesis_result.get("cross_algorithm_insights", {})
                if isinstance(insights, dict) and "rankings" in insights:
                    f.write("| Rank | Model | Best Acc | Mean Acc | Std Dev | Trials |\n")
                    f.write("|------|-------|----------|----------|---------|--------|\n")
                    for i, r in enumerate(insights["rankings"][:10], 1):
                        f.write(
                            f"| {i} | {r['model']} | {r['best_accuracy']:.2%} | {r['mean_accuracy']:.2%} | {r['std']:.4f} | {r['trials']} |\n")
                    f.write("\n")
                else:
                    f.write(f"{insights}\n\n")

                # Task-Specific Winners
                f.write("## 📊 Task-Specific Winners\n\n")
                task_winners = synthesis_result.get("task_specific_winners", {})
                if isinstance(task_winners, dict):
                    for task, winners in task_winners.items():
                        f.write(f"### {task.replace('_', ' ').title()}\n")
                        for i, w in enumerate(winners, 1):
                            f.write(
                                f"{i}. **{w['model']}**: {w['accuracy']:.2%} ({w['params']:,} params)\n")
                        f.write("\n")

                # Efficiency Analysis
                f.write("## ⚡ Efficiency Analysis\n\n")
                efficiency = synthesis_result.get("efficiency_analysis", {})

                if "top_epoch_efficient" in efficiency:
                    f.write("### Top Models by Epoch Efficiency (Accuracy / Epoch)\n")
                    f.write(
                        "*Models that converge fastest - high accuracy with fewer epochs.*\n\n")
                    f.write("| Model | Task | Accuracy | Epochs | Acc/Epoch |\n")
                    f.write("|-------|------|----------|--------|----------|\n")
                    for r in efficiency["top_epoch_efficient"][:5]:
                        eff = r['epoch_efficiency']
                        f.write(
                            f"| {r['model_name']} | {r['task_name']} | {r['accuracy']:.2%} | {r['num_epochs']} | {eff:.4f} |\n")
                    f.write("\n")

                if "top_param_efficient" in efficiency:
                    f.write("### Top Models by Parameter Efficiency (Accuracy / M-Params)\n")
                    f.write("*Models that achieve high performance with fewer parameters.*\n\n")
                    for r in efficiency["top_param_efficient"][:5]:
                        params_m = r['param_count'] / 1e6
                        f.write(
                            f"- **{r['model_name']}**: {r['accuracy']:.2%} with {params_m:.2f}M params (efficiency: {r['param_efficiency']:.2f})\n")
                    f.write("\n")

                f.write("## ⚠️ Failure Analysis\n")
                fails = synthesis_result.get("failure_analysis", {})
                if isinstance(fails, dict):
                    if "patterns" in fails and fails["patterns"]:
                        f.write("\n**Detected Patterns:**\n")
                        for p in fails["patterns"]:
                            f.write(f"- {p}\n")
                        f.write("\n")
                    if "counts" in fails:
                        f.write("\n**Failure Counts:**\n")
                        for k, v in fails["counts"].items():
                            f.write(f"- **{k}**: {v} failures\n")
                else:
                    f.write(f"{fails}\n\n")

                f.write("\n## 💡 Quick Wins & Suggestions\n\n")
                wins = synthesis_result.get("quick_wins", [])
                if isinstance(wins, list) and wins:
                    for win in wins:
                        f.write(f"- {win}\n")
                else:
                    f.write("All systems running smoothly. Continue exploration.\n")

                f.write("\n## 🔬 Research Gaps\n\n")
                gaps = synthesis_result.get("research_gaps", [])
                if isinstance(gaps, list) and gaps:
                    for gap in gaps:
                        f.write(f"- {gap}\n")
                else:
                    f.write("No major research gaps identified.\n")

            logger.info("✓ Research synthesis generated (synthesis/)")
        except Exception as e:
            logger.warning("No trajectories found for synthesis.")
        except Exception as e:
            logger.error(f"Failed to generate synthesis: {e}", exc_info=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Reports saved to: {report_path}")
        logger.info(f"  - index.md: Main comprehensive report")
        logger.info(f"  - images/: Visualizations and ML analysis")
        logger.info(f"  - report.tex: LaTeX source (compile with ./compile_report.sh)")
        logger.info(f"  - synthesis/: High-level strategic insights")
        logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument("--dir", default="reports", help="Output directory for reports")
    args = parser.parse_args()

    scientist = AutoScientist()

    if args.report:
        scientist.generate_reports(args.dir)
    else:
        scientist.run()
