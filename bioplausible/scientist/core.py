"""
AutoScientist: The Autonomous Discovery Agent.

This module implements the core logic for the continuous experiment runner.
It manages the experiment lifecycle:
1. State Analysis: What have we learned so far?
2. Strategy: What should we do next? (Smoke -> Shallow -> Standard -> Deep)
3. Execution: Run the experiment.
4. Learning: Update the knowledge base.
"""

import gc
import json
import logging
import random
import signal
import time
from typing import Any, Dict, Optional, Tuple

import optuna  # noqa: F401
import torch

from bioplausible.hyperopt import (
    PatientLevel,
    create_constrained_optuna_config,
    get_evaluation_config,
)
from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.scientist.dashboard import DASHBOARD
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

# Configure Logging to File ONLY (Dashboard handles stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scientist.log")],
)
logger = logging.getLogger("AutoScientist")

DB_PATH = "bioplausible.db"


class AutoScientist:
    """
    The Autonomous Scientist Agent.

    This agent runs in a continuous loop, analyzing previous results,
    planning new experiments, and executing them.
    """

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(
        self,
        db_path: str = DB_PATH,
        task_filter: Optional[str] = None,
        tier_limit: Optional[str] = None,
    ):
        self.db_path = db_path
        self.state = ExperimentState(db_path)
        self.decision_logger = DecisionLogger(db_path)
        self.strategy = ScientistStrategy(
            self.state,
            self.decision_logger,
            task_filter=task_filter,
            tier_limit=tier_limit,
        )
        self.resources = ResourceMonitor()
        self.running = True
        self.consecutive_failures = 0

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig: int, frame: Any) -> None:
        logger.info("Interrupt received. Finishing current trial...")
        self.running = False

    def run(self) -> None:
        """
        Start the continuous discovery loop.
        """
        logger.info("AutoScientist initialized. Starting continuous discovery...")
        DASHBOARD.start()
        DASHBOARD.log("AutoScientist Started", style="bold green")

        try:
            self._run_discovery_loop()
        finally:
            DASHBOARD.stop()
            logger.info("AutoScientist shutting down. Cleaning up...")
            self.state.close()
            logger.info("Shutdown complete.")

    def _run_discovery_loop(self) -> None:
        """
        The main loop execution logic.
        """
        while self.running:
            DASHBOARD.update()

            if self._check_resources_pause():
                continue

            if self._check_failures_pause():
                continue

            task = self.strategy.plan_next()
            if not self._handle_no_task(task):
                continue

            self._log_task_start(task)

            try:
                metrics = self._process_task(task)
                self._handle_result(metrics, task)
            except Exception as e:
                self._handle_error(e)

            # Post-trial cleanup
            time.sleep(1)
            self._cleanup_memory()

    def _check_resources_pause(self) -> bool:
        """Check if resources are exhausted and pause if necessary."""
        if self.resources.should_pause():
            DASHBOARD.log("Resources exhausted. Pausing...", style="yellow")
            time.sleep(60)
            return True
        return False

    def _check_failures_pause(self) -> bool:
        """Check if too many consecutive failures have occurred."""
        if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            DASHBOARD.log(
                f"Too many failures ({self.consecutive_failures}). Sleeping 5m.",
                style="bold red",
            )
            time.sleep(300)
            self.consecutive_failures = 0
            # Don't return True, just reset and continue, or could return True to skip planning
        return False

    def _handle_no_task(self, task: Optional[ExperimentTask]) -> bool:
        """Handle the case where no task is available."""
        if not task:
            DASHBOARD.log("No viable experiments. Sleeping 60s...")
            time.sleep(60)
            return False
        return True

    def _log_task_start(self, task: ExperimentTask) -> None:
        """Log the start of a task."""
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

        msg = f"Starting {type_str}: {task.model_name} | {task.task_name} | {task.tier.name}"
        logger.info(msg)
        DASHBOARD.log(msg, style="blue")

    def _process_task(self, task: ExperimentTask) -> Optional[Dict[str, float]]:
        """
        Prepare configuration and execute the task.
        Returns metrics if successful, None otherwise.
        """
        # Load Optuna Study
        study = self.state.get_optuna_study(task.study_name)

        trial = None
        if task.fixed_config:
            config, job_id = self._prepare_fixed_config(task)
        else:
            trial, config, job_id = self._prepare_optuna_config(task, study)

        # Inject Tier Config
        self._inject_tier_config(config, task)

        # Identify interesting params for logging
        self._update_dashboard_with_config(config, job_id, task)

        if task.is_robustness_check:
            metrics = self._execute_robustness_check(task, config)
        else:
            metrics = self._execute_standard_trial(task, config, trial, job_id)

        # Update Optuna if trial exists
        if trial:
            if metrics:
                acc = metrics.get("accuracy", 0.0)
                study.tell(trial, acc)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

        return metrics

    def _prepare_fixed_config(self, task: ExperimentTask) -> Tuple[Dict[str, Any], str]:
        """Prepare configuration for fixed tasks."""
        config = task.fixed_config.copy()  # type: ignore

        # Ensure fold is set for CV
        if task.fold_index is not None:
            config["fold"] = task.fold_index

        # Determine Job ID
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

        return config, job_id

    def _prepare_optuna_config(
        self, task: ExperimentTask, study: optuna.Study
    ) -> Tuple[optuna.trial.Trial, Dict[str, Any], str]:
        """Prepare configuration using Optuna."""
        # Warm-Start Logic
        self._attempt_warm_start(study, task)

        trial = study.ask()

        # Pass dynamic constraints (intelligence)
        constraints = {}
        if task.constraints:
            constraints.update(task.constraints)
            logger.info(f"  > Applying intelligent constraints: {constraints}")

        config = create_constrained_optuna_config(
            trial,
            task.model_name,
            custom_constraints=constraints,
            task_name=task.task_name,
        )

        # Determine Job ID
        if trial.number is not None:
            job_id = str(trial.number)
        elif hasattr(trial, "_trial_id"):
            job_id = str(trial._trial_id)
        else:
            job_id = "Unknown"

        # Log metadata for reports
        trial.set_user_attr("model_name", task.model_name)
        trial.set_user_attr("task_name", task.task_name)
        trial.set_user_attr("tier", task.tier.value)

        return trial, config, job_id

    def _attempt_warm_start(self, study: optuna.Study, task: ExperimentTask) -> None:
        """Attempt to warm-start the study from best previous trials."""
        if random.random() < 0.2:  # 20% chance to warm start
            try:
                if len(study.trials) > 0:
                    best_trial = study.best_trial
                    if best_trial:
                        logger.info(
                            f"  > Warm-starting from Trial #{best_trial.number} (Acc: {best_trial.value:.2%})"
                        )
                        study.enqueue_trial(best_trial.params)
            except Exception as e:
                logger.warning(f"Warm start failed: {e}")

    def _inject_tier_config(self, config: Dict[str, Any], task: ExperimentTask) -> None:
        """Inject tier-specific configuration and metadata."""
        tier_config = get_evaluation_config(task.tier)
        config["epochs"] = tier_config.epochs
        config["batch_size"] = tier_config.batch_size
        config["early_stopping_patience"] = 3

        # Metadata
        config["tier"] = task.tier.value
        config["task"] = task.task_name
        config["model"] = task.model_name

        if task.fixed_config:
            config["is_verification"] = True
            config["verified_trial_id"] = task.verification_of_trial_id

        if task.is_robustness_check:
            config["is_robustness_check"] = True

        if task.is_ablation:
            config["is_ablation"] = True
            config["ablation_param"] = task.ablation_param
            config["save_artifacts"] = True

        if task.is_transfer:
            config["is_transfer"] = True
            config["transfer_from"] = task.transfer_from_trial
            config["save_artifacts"] = True

        if task.is_continual:
            config["is_continual"] = True
            config["continual_step"] = task.continual_step
            config["save_artifacts"] = True
            if task.transfer_from_trial:
                config["transfer_from"] = task.transfer_from_trial

    def _update_dashboard_with_config(
        self, config: Dict[str, Any], job_id: str, task: ExperimentTask
    ) -> None:
        """Update the dashboard with the current trial configuration."""
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
        interesting_params = {k: v for k, v in config.items() if k not in ignore_keys}
        DASHBOARD.set_trial(
            job_id, task.model_name, task.task_name, task.tier.name, interesting_params
        )

    def _handle_result(
        self, metrics: Optional[Dict[str, float]], task: ExperimentTask
    ) -> None:
        """Handle the result of a trial execution."""
        if metrics:
            acc = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", float("inf"))
            DASHBOARD.log(f"Result: Acc={acc:.2%}, Loss={loss:.4f}", style="bold green")
            DASHBOARD.complete_trial("completed", acc)
            self.consecutive_failures = 0
        else:
            DASHBOARD.log("Trial failed.", style="bold red")
            DASHBOARD.complete_trial("failed", 0.0)
            self.consecutive_failures += 1

    def _handle_error(self, e: Exception) -> None:
        """Handle exceptions during trial execution."""
        logger.error(f"Error executing trial: {e}", exc_info=True)
        DASHBOARD.log(f"Error: {e}", style="bold red")
        self.consecutive_failures += 1
        time.sleep(5)

    def _cleanup_memory(self) -> None:
        """Aggressively cleanup memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _execute_robustness_check(
        self, task: ExperimentTask, config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Run robustness suite and return metrics."""
        DASHBOARD.log("Running Robustness Suite...")

        # Note: weights_path logic was stubbed in original, kept simple here
        weights_path = None

        score = run_robustness_check(
            task.model_name, task.task_name, config, weights_path=weights_path
        )
        return {
            "accuracy": score,
            "loss": 0.0,
            "robustness_score": score,
            "time": 0.0,
            "param_count": 0.0,
        }

    def _execute_standard_trial(
        self,
        task: ExperimentTask,
        config: Dict[str, Any],
        trial: Optional[optuna.trial.Trial],
        job_id: str,
    ) -> Optional[Dict[str, float]]:
        """Run a standard training trial."""
        quick = task.tier == PatientLevel.SMOKE

        if trial:
            trial.set_user_attr("config", json.dumps(config))

        config["job_id"] = job_id

        return run_single_trial_task(
            task=task.task_name,
            model_name=task.model_name,
            config=config,
            storage_path=DB_PATH,
            quick_mode=quick,
        )

    def generate_reports(self, output_dir: str = "reports") -> None:
        """
        Generates comprehensive Scientist++ reports with ML analysis, visualizations,
        statistical tests, and high-level synthesis insights.
        """
        try:
            from bioplausible.scientist.report.orchestrator import ReportOrchestrator

            orchestrator = ReportOrchestrator(self.db_path, output_dir)
            orchestrator.generate_reports()
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument(
        "--dir", default="reports", help="Output directory for reports"
    )
    parser.add_argument(
        "--tier-limit",
        type=str,
        default=None,
        help="Limit maximum tier (smoke, shallow, standard, deep)",
    )
    args = parser.parse_args()

    scientist = AutoScientist(tier_limit=args.tier_limit)

    if args.report:
        scientist.generate_reports(args.dir)
    else:
        scientist.run()
