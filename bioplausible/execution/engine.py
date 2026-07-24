"""
AutoScientist: The Autonomous Discovery Agent.

This module implements the core logic for the continuous experiment runner.
It manages the experiment lifecycle:
1. State Analysis: What have we learned so far?
2. Strategy: What should we do next? (Smoke -> Shallow -> Standard -> Deep)
3. Execution: Run the experiment.
4. Learning: Update the knowledge base.
"""

import contextlib
import gc
import json
import logging
import random
import shutil
import signal
import tempfile
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import torch

from bioplausible.execution.dashboard import DASHBOARD
from bioplausible.execution.decisions import DecisionLogger
from bioplausible.execution.failure_tracker import FailureRecord
from bioplausible.execution.resources import ResourceMonitor
from bioplausible.execution.robustness import run_robustness_check
from bioplausible.execution.state import ExperimentState
from bioplausible.execution.strategy import ExecutionStrategy
from bioplausible.execution.task import ExperimentTask
from bioplausible.hyperopt import (
    PatientLevel,
    create_constrained_optuna_config,
    get_evaluation_config,
)
from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.hyperopt.parallel_runner import ParallelTrialRunner
from bioplausible.lightning_.experiment import run_pl_trial

# Re-export for backward compatibility
__all__ = [
    "ExecutionEngine",
    "ExecutionStrategy",
    "ExperimentState",
    "ExperimentTask",
    "ResourceMonitor",
]

# Configure Logging to File ONLY (Dashboard handles stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scientist.log")],
)
logger = logging.getLogger("AutoScientist")

DB_PATH = "bioplausible.db"


class ExecutionEngine:
    """
    The Autonomous Scientist Agent (Execution Engine).

    This agent runs in a continuous loop, analyzing previous results,
    planning new experiments, and executing them.

    Note: This is the execution engine. For the LLM meta-reasoner,
    see AutoScientistCampaign in bioplausible.autoscientist.
    """

    MAX_CONSECUTIVE_FAILURES = 5
    MAX_RETRIES = 3
    CIRCUIT_BREAKER_THRESHOLD = 10
    CIRCUIT_BREAKER_RESET_INTERVAL = 300  # 5 minutes

    def __init__(
        self,
        db_path: str = DB_PATH,
        task_filter: str | None = None,
        tier_limit: str | None = None,
        num_workers: int = 1,
        report_interval: int = 50,
    ):
        self.db_path = db_path
        self.num_workers = num_workers
        self.report_interval = report_interval
        self.trial_count = 0
        self.last_report_trial = 0
        self.state = ExperimentState(db_path)
        self.decision_logger = DecisionLogger(db_path)
        self.strategy = ExecutionStrategy(
            self.state,
            self.decision_logger,
            task_filter=task_filter,
            tier_limit=tier_limit,
        )
        self.resources = ResourceMonitor()
        self.running = True
        self.consecutive_failures = 0

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_tripped_at = 0.0
        self._circuit_failure_count = 0
        self._model_failure_counts: dict[str, int] = {}

        self.parallel_runner = None
        if num_workers > 1:
            self.parallel_runner = ParallelTrialRunner(num_workers, db_path)

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig: int, frame: Any) -> None:
        logger.info("Interrupt received. Finishing current trial...")
        self.running = False

    def _print_resume_context(self) -> None:
        """
        Print context when resuming from a previous session.
        """
        progress = self.state.get_progress()
        total_trials = sum(
            sum(tier.get("count", 0) for tier in task.values())
            for model in progress.values()
            for task in model.values()
        )

        print("\n" + "=" * 60)
        print("📋 RESUME CONTEXT")
        print("=" * 60)
        print(f"Total trials completed: {total_trials}")

        recent_models = self.state.get_recent_models(limit=5)
        recent_tasks = self.state.get_recent_tasks(limit=5)
        if recent_models:
            print(f"Recent models: {', '.join(recent_models)}")
        if recent_tasks:
            print(f"Recent tasks: {', '.join(recent_tasks)}")

        failure_analysis = self.state.get_failure_analysis()
        if failure_analysis.get("patterns"):
            print("\n⚠️ Known failure patterns:")
            for p in failure_analysis["patterns"][:2]:
                print(f"  - {p}")

        print("=" * 60 + "\n")

        logger.info(f"Resuming from trial #{total_trials}")

    def run(self) -> None:
        """
        Start the continuous discovery loop.
        """
        logger.info("AutoScientist initialized. Starting continuous discovery...")
        DASHBOARD.start()
        DASHBOARD.log("AutoScientist Started", style="bold green")
        DASHBOARD.set_system_status("Active", "bold green")

        self._print_resume_context()

        try:
            self._run_discovery_loop()
        finally:
            DASHBOARD.stop()
            logger.info("AutoScientist shutting down. Cleaning up...")
            self.state.close()
            logger.info("Shutdown complete.")

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (too many recent failures)."""
        if not self._circuit_open:
            return False
        elapsed = time.time() - self._circuit_tripped_at
        if elapsed > self.CIRCUIT_BREAKER_RESET_INTERVAL:
            self._circuit_open = False
            self._circuit_failure_count = 0
            logger.info("Circuit breaker reset after cooldown period")
            DASHBOARD.log("Circuit breaker reset - resuming operations", style="green")
            return False
        return True

    def _get_wait_time(self, attempt: int) -> float:
        """Exponential backoff: 2^attempt seconds, capped at 60."""
        return min(2.0**attempt, 60.0)

    def _classify_failure(self, error: Exception) -> str:
        """Classify failure type to determine retry strategy."""
        error_str = str(error).lower()
        if any(
            term in error_str
            for term in ["timeout", "connection", "broken pipe", "dataloader"]
        ):
            return "transient"
        if any(term in error_str for term in ["oom", "out of memory", "cuda out of"]):
            return "resource"
        if any(term in error_str for term in ["nan", "inf", "exploding"]):
            return "instability"
        return "permanent"

    def _run_discovery_loop(self) -> None:
        """
        The main loop execution logic.
        """
        while self.running:
            DASHBOARD.update()

            if self._check_circuit_breaker():
                continue

            if self._check_resources_pause():
                continue

            if self._check_failures_pause():
                continue

            # Check for periodic reporting
            if (
                self.trial_count > 0
                and (self.trial_count - self.last_report_trial) >= self.report_interval
            ):
                logger.info("Generating periodic research reports...")
                DASHBOARD.log("Generating Periodic Reports...", style="cyan")
                try:
                    self.generate_reports()
                    self.last_report_trial = self.trial_count
                except Exception as e:
                    logger.error(f"Periodic reporting failed: {e}")

            if self.num_workers > 1 and self.parallel_runner:
                # Parallel Execution
                tasks = self.strategy.plan_batch(self.num_workers)
                if not tasks:
                    self._handle_no_task(None)
                    continue

                logger.info(
                    "Starting batch of %d tasks with %d workers.",
                    len(tasks),
                    self.num_workers,
                )

                try:
                    # Resolve configs first
                    configs = []
                    for t in tasks:
                        # We need to resolve configs here to pass to runner
                        # This duplicates logic in _process_task a bit, but necessary
                        study = self.state.get_optuna_study(t.study_name)

                        if t.fixed_config:
                            conf, _ = self._prepare_fixed_config(t)
                        else:
                            # Parallel Optuna sampling; SQLite handles it.
                            _, conf, _ = self._prepare_optuna_config(t, study)

                        self._inject_tier_config(conf, t)
                        configs.append(conf)

                    results = self.parallel_runner.run_batch(tasks, configs)

                    for i, metrics in enumerate(results):
                        self._handle_result(metrics, tasks[i])

                    self.trial_count += len(results)

                except Exception as e:
                    logger.error(f"Parallel batch failed: {e}", exc_info=True)
                    self.consecutive_failures += 1
            else:
                # Sequential Execution
                task = self.strategy.plan_next()
                if not self._handle_no_task(task):
                    continue

                self._log_task_start(task)

                # Attempt with retry logic
                metrics = self._process_with_retry(task)
                if metrics is not None:
                    self._handle_result(metrics, task)
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                self.trial_count += 1

            # Post-trial cleanup
            time.sleep(1)
            self._cleanup_memory()

    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state and pause if open."""
        if self._is_circuit_open():
            remaining = int(
                self.CIRCUIT_BREAKER_RESET_INTERVAL
                - (time.time() - self._circuit_tripped_at)
            )
            logger.warning(f"Circuit breaker open. Cooling down for {remaining}s")
            DASHBOARD.set_system_status(
                f"Circuit Breaker - Cooldown {remaining}s", "yellow"
            )
            for i in range(min(remaining, 10), 0, -1):
                DASHBOARD.set_system_status(
                    f"Circuit Breaker - Cooldown {i}s", "yellow"
                )
                time.sleep(1)
            return True
        return False

    def _check_resources_pause(self) -> bool:
        """Check if resources are exhausted and pause if necessary."""
        if self.resources.should_pause():
            DASHBOARD.log("Resources exhausted. Pausing...", style="yellow")
            for i in range(60, 0, -1):
                DASHBOARD.set_system_status(f"Paused - Retry in {i}s", "yellow")
                time.sleep(1)
            return True
        DASHBOARD.set_system_status("Active", "bold green")
        return False

    def _check_failures_pause(self) -> bool:
        """Check if too many consecutive failures have occurred."""
        if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            DASHBOARD.log(
                f"Too many consecutive failures"
                f" ({self.consecutive_failures}). Triggering Safe Mode...",
                style="bold red",
            )
            DASHBOARD.set_system_status("Safe Mode (Diagnostic)", "bold yellow")

            # Run Diagnostic
            success = self._run_diagnostic_task()
            if success:
                DASHBOARD.log(
                    "Diagnostic Passed. Resuming operations.", style="bold green"
                )
                DASHBOARD.set_system_status("Active", "bold green")
                self.consecutive_failures = 0
                return False
            else:
                logger.critical("Diagnostic Failed! System unstable. Terminating.")
                DASHBOARD.log(
                    "CRITICAL: Diagnostic Failed. Terminating Agent.", style="bold red"
                )
                DASHBOARD.set_system_status("Terminated", "bold red")
                self.running = False
                return True

        return False

    def _run_diagnostic_task(self) -> bool:
        """Run a simple diagnostic task to check system health."""
        # Create a simple task (Backprop on Digits is very fast/stable)
        task = ExperimentTask(
            model_name="Backprop Baseline",
            task_name="digits",
            tier=PatientLevel.SMOKE,
            study_name="diagnostic",
            priority=1000.0,
            fixed_config={"epochs": 1, "batch_size": 32, "hidden_dim": 16},
        )
        try:
            DASHBOARD.log("Running Diagnostic Task (Digits/MLP)...", style="yellow")
            metrics = self._process_task(task)
            return metrics is not None
        except Exception as e:
            logger.error(f"Diagnostic failed: {e}")
            return False

    def _handle_no_task(self, task: ExperimentTask | None) -> bool:
        """Handle the case where no task is available."""
        if not task:
            DASHBOARD.log("No viable experiments. Sleeping 60s...")
            for i in range(60, 0, -1):
                DASHBOARD.set_system_status(
                    f"Waiting for Tasks - Retry in {i}s", "yellow"
                )
                time.sleep(1)
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

        msg = (
            f"Starting {type_str}: {task.model_name}"
            f" | {task.task_name} | {task.tier.name}"
        )
        logger.info(msg)
        DASHBOARD.log(msg, style="blue")

    def _process_with_retry(self, task: ExperimentTask) -> dict[str, float] | None:
        """Process a task with retry logic and exponential backoff.

        Attempts up to MAX_RETRIES for transient/resource failures.
        Permanent failures are not retried.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                if attempt > 0:
                    wait = self._get_wait_time(attempt)
                    logger.info(
                        "Retry %d/%d for %s/%s in %.1fs",
                        attempt,
                        self.MAX_RETRIES,
                        task.model_name,
                        task.task_name,
                        wait,
                    )
                    DASHBOARD.log(
                        f"Retry {attempt}/{self.MAX_RETRIES} in {wait:.1f}s",
                        style="yellow",
                    )
                    time.sleep(wait)

                return self._process_task(task)

            except Exception as e:
                failure_type = self._classify_failure(e)

                self.state.failure_tracker.log_failure(
                    FailureRecord(
                        timestamp=datetime.now().isoformat(),
                        model_name=task.model_name,
                        task_name=task.task_name,
                        tier=task.tier.value,
                        trial_id=None,
                        failure_type=failure_type,
                        failure_epoch=None,
                        failure_batch=None,
                        config={},
                        last_metrics={},
                        stack_trace=traceback.format_exc(),
                    )
                )

                if failure_type == "permanent":
                    logger.error(
                        f"Permanent failure for {task.model_name}/{task.task_name}: {e}"
                    )
                    DASHBOARD.log(f"Permanent failure: {e}", style="bold red")
                    break

                if failure_type == "instability":
                    # Track per-model instability
                    key = f"{task.model_name}/{task.task_name}"
                    self._model_failure_counts[key] = (
                        self._model_failure_counts.get(key, 0) + 1
                    )
                    if self._model_failure_counts[key] >= 3:
                        logger.warning(
                            "Model %s unstable on %s, blacklisting temporarily",
                            task.model_name,
                            task.task_name,
                        )
                        DASHBOARD.log(
                            f"Model {task.model_name} unstable, skipping future trials",
                            style="red",
                        )
                        break

                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Transient failure (attempt {attempt + 1}): {e}")
                else:
                    logger.error(
                        "All %d retries exhausted for %s/%s: %s",
                        self.MAX_RETRIES,
                        task.model_name,
                        task.task_name,
                        e,
                    )
                    DASHBOARD.log(f"All retries exhausted: {e}", style="bold red")
                    # Trip circuit breaker on repeated failures
                    self._circuit_failure_count += 1
                    if self._circuit_failure_count >= self.CIRCUIT_BREAKER_THRESHOLD:
                        self._circuit_open = True
                        self._circuit_tripped_at = time.time()
                        logger.critical("Circuit breaker tripped! Too many failures.")
                        DASHBOARD.log("CIRCUIT BREAKER TRIPPED", style="bold red")

        return None

    def _run_asi_evolve(self, task: ExperimentTask) -> dict[str, float] | None:
        """
        ASI-Evolve integration removed in REFACTOR2 (asi_evolve/ package deleted).
        """
        logger.warning(
            "ASI-Evolve integration removed. Skipping evolve task: %s",
            task.study_name,
        )
        DASHBOARD.log(
            f"ASI-Evolve no longer available (task: {task.study_name})",
            style="bold yellow",
        )
        return None

    def _process_task(self, task: ExperimentTask) -> dict[str, float] | None:
        """
        Prepare configuration and execute the task.
        Returns metrics if successful, None otherwise.
        """
        if task.is_evolve:
            return self._run_asi_evolve(task)

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

                # Save extended metrics (like robustness scores)
                for k, v in metrics.items():
                    if k not in ["accuracy", "loss"] and isinstance(
                        v, (int, float, str)
                    ):
                        trial.set_user_attr(k, v)

                study.tell(trial.number, acc)
            else:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

        return metrics

    def _prepare_fixed_config(self, task: ExperimentTask) -> tuple[dict[str, Any], str]:
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
    ) -> tuple[optuna.trial.Trial, dict[str, Any], str]:
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
                            "  > Warm-starting from Trial #%d (Acc: %.2%%)",
                            best_trial.number,
                            best_trial.value,
                        )
                        study.enqueue_trial(best_trial.params)
            except Exception as e:
                logger.warning(f"Warm start failed: {e}")

    def _inject_tier_config(self, config: dict[str, Any], task: ExperimentTask) -> None:
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
        self, config: dict[str, Any], job_id: str, task: ExperimentTask
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
            str(job_id),
            task.model_name,
            task.task_name,
            task.tier.name,
            interesting_params,
        )

    def _handle_result(
        self, metrics: dict[str, float] | None, task: ExperimentTask
    ) -> None:
        """Handle the result of a trial execution."""
        if metrics:
            acc = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", float("inf"))
            DASHBOARD.log(f"Result: Acc={acc:.2%}, Loss={loss:.4f}", style="bold green")
            DASHBOARD.complete_trial("completed", metrics)
            self.consecutive_failures = 0
        else:
            DASHBOARD.log("Trial failed.", style="bold red")
            DASHBOARD.complete_trial("failed", {"accuracy": 0.0})
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
        self, task: ExperimentTask, config: dict[str, Any]
    ) -> dict[str, float]:
        """Run robustness suite and return metrics."""
        DASHBOARD.log("Running Robustness Suite...")

        ctx = contextlib.nullcontext()
        if task.verification_of_trial_id:
            ctx = self._get_weights_context(task.verification_of_trial_id)

        # Determine output directory for interpretability artifacts
        output_dir = None
        if task.verification_of_trial_id:
            output_dir = (
                f"artifacts/trial_{task.verification_of_trial_id}/interpretability"
            )

        with ctx as weights_path:
            metrics = run_robustness_check(
                task.model_name,
                task.task_name,
                config,
                weights_path=weights_path,
                output_dir=output_dir,
            )

        score = metrics.get("robustness_score", 0.0)

        result = {
            "accuracy": score,
            "loss": metrics.get("loss", 0.0),
            "time": metrics.get("time", 0.0),
            "param_count": metrics.get("param_count", 0.0),
        }
        for k, v in metrics.items():
            if k not in result:
                result[k] = v
        return result

    @contextlib.contextmanager
    def _get_weights_context(self, trial_id: int):
        """
        Context manager to find and yield path to weights for a trial.
        Handles extraction from zip artifacts if necessary.
        """
        artifact_dir = Path("artifacts")
        found_path = None
        temp_dir = None

        if artifact_dir.exists():
            # 1. Check directories
            for item in artifact_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"trial_{trial_id}_"):
                    p = item / "model.pt"
                    if p.exists():
                        found_path = str(p)
                        break

            # 2. Check zips if not found
            if not found_path:
                for item in artifact_dir.iterdir():
                    if item.suffix == ".zip" and item.name.startswith(
                        f"trial_{trial_id}_"
                    ):
                        temp_dir = tempfile.mkdtemp()
                        try:
                            with zipfile.ZipFile(item, "r") as zf:
                                zf.extract("model.pt", temp_dir)
                                found_path = str(Path(temp_dir) / "model.pt")
                        except Exception as e:
                            logger.warning(f"Failed to extract artifact: {e}")
                        break

        try:
            yield found_path
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir)

    def _execute_standard_trial(
        self,
        task: ExperimentTask,
        config: dict[str, Any],
        trial: optuna.trial.Trial | None,
        job_id: str,
    ) -> dict[str, float] | None:
        """Run a standard training trial. Can use Lightning if configured."""
        quick = task.tier == PatientLevel.SMOKE

        if trial:
            trial.set_user_attr("config", json.dumps(config))

        config["job_id"] = job_id

        use_pl = config.get("use_lightning", False)

        if use_pl:
            return run_pl_trial(
                model_name=task.model_name,
                optimizer_name=config.get("optimizer", "adam"),
                config=config,
                train_loader=self._get_train_loader(task),
                val_loader=self._get_val_loader(task),
                quick_mode=quick,
            )

        return run_single_trial_task(
            task=task.task_name,
            model_name=task.model_name,
            config=config,
            storage_path=DB_PATH,
            quick_mode=quick,
        )

    def _get_train_loader(self, task: ExperimentTask):
        """Get training DataLoader for a task."""
        from bioplausible.datasets import create_data_loaders

        batch_size = (
            task.fixed_config.get("batch_size", 64) if task.fixed_config else 64
        )

        train_loader, _ = create_data_loaders(
            dataset_name=task.task_name,
            batch_size=batch_size,
            flatten=True,
        )
        return train_loader

    def _get_val_loader(self, task: ExperimentTask):
        """Get validation DataLoader for a task."""
        from bioplausible.datasets import create_data_loaders

        batch_size = (
            task.fixed_config.get("batch_size", 64) if task.fixed_config else 64
        )

        _, val_loader = create_data_loaders(
            dataset_name=task.task_name,
            batch_size=batch_size,
            flatten=True,
        )
        return val_loader

    def generate_reports(self, output_dir: str = "reports") -> None:
        """
        Generates comprehensive Scientist++ reports with ML analysis, visualizations,
        statistical tests, and high-level synthesis insights.
        """
        try:
            from bioplausible.execution.report.orchestrator import ReportOrchestrator

            orchestrator = ReportOrchestrator(self.db_path, output_dir)
            orchestrator.generate_reports()
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument("--dir", default="reports", help="Output directory for reports")
    parser.add_argument(
        "--tier-limit",
        type=str,
        default=None,
        help="Limit maximum tier (smoke, shallow, standard, deep)",
    )
    args = parser.parse_args()

    engine = ExecutionEngine(tier_limit=args.tier_limit)

    if args.report:
        engine.generate_reports(args.dir)
    else:
        engine.run()


# Backward compatibility alias
AutoScientist = ExecutionEngine
