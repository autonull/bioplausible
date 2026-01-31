"""
AutoScientist: The Autonomous Discovery Agent.

This module implements the core logic for the continuous experiment runner.
It manages the experiment lifecycle:
1. State Analysis: What have we learned so far?
2. Strategy: What should we do next? (Smoke -> Shallow -> Standard -> Deep)
3. Execution: Run the experiment.
4. Learning: Update the knowledge base.
"""

import time
import signal
import sys
import random
import optuna
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# psutil needed for resource monitoring
try:
    import psutil
except ImportError:
    psutil = None

from bioplausible.models.registry import MODEL_REGISTRY, ModelSpec
from bioplausible.hyperopt import create_optuna_space, PatientLevel, get_evaluation_config
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.scientist.robustness import run_robustness_check

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scientist.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AutoScientist")

DB_PATH = "bioplausible.db"

@dataclass
class ExperimentTask:
    """Represents a single planned experiment."""
    model_name: str
    task_name: str
    tier: PatientLevel
    study_name: str
    priority: float  # Higher is better
    fixed_config: Optional[Dict[str, Any]] = None # If set, run this config exactly (verification)
    verification_of_trial_id: Optional[int] = None
    fold_index: Optional[int] = None # For Cross-Validation (0-4)
    last_run_timestamp: Optional[str] = None
    is_robustness_check: bool = False

class ResourceMonitor:
    """Monitors system resources to prevent overload."""
    def __init__(self, cpu_limit=90.0, mem_limit=90.0):
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit

    def should_pause(self) -> bool:
        if not psutil: return False

        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent

        if cpu > self.cpu_limit or mem > self.mem_limit:
            logger.warning(f"System Load High: CPU={cpu}%, Mem={mem}%. Pausing...")
            return True
        return False

class ExperimentState:
    """
    Analyzes the current state of research by querying the database.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.storage = HyperoptStorage(db_path)

    def get_progress(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns a nested dictionary with stats.
        """
        trials = self.storage.get_all_trials()
        progress = {}

        for t in trials:
            if t.status != "completed":
                continue

            model = t.model_name
            task = t.config.get("task")
            tier_val = t.config.get("tier")

            if not task or not tier_val: continue

            if model not in progress: progress[model] = {}
            if task not in progress[model]: progress[model][task] = {}
            if tier_val not in progress[model][task]:
                progress[model][task][tier_val] = {
                    'count': 0, 'best_acc': -1.0, 'trials': [], 'last_run_ts': 0.0
                }

            entry = progress[model][task][tier_val]
            entry['count'] += 1
            entry['trials'].append(t)

            if t.accuracy > entry['best_acc']:
                entry['best_acc'] = t.accuracy

        return progress

    def get_optuna_study(self, study_name: str):
        """Load or create an Optuna study."""
        return optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{self.db_path}",
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler()
        )

    def close(self):
        self.storage.close()


class ScientistStrategy:
    """
    The Brains. Decides what to run next.
    """

    CRITERIA = {
        PatientLevel.SMOKE: lambda acc: acc > 0.15,
        PatientLevel.SHALLOW: lambda acc: acc > 0.40,
        PatientLevel.STANDARD: lambda acc: acc > 0.60,
        PatientLevel.CROSS_VAL: lambda acc: True, # CV just needs to run 5 times
        PatientLevel.DEEP: lambda acc: acc > 0.80, # Deep bar
    }

    TASKS = ["vision", "lm", "rl"]

    def __init__(self, state: ExperimentState):
        self.state = state

    def plan_next(self) -> Optional[ExperimentTask]:
        """
        Scans all possibilities and returns the highest priority experiment.
        """
        progress = self.state.get_progress()
        candidates = []

        for spec in MODEL_REGISTRY:
            tasks = spec.task_compat if spec.task_compat else self.TASKS

            for task in tasks:
                # 1. SMOKE
                smoke_stats = self._get_stats(progress, spec.name, task, PatientLevel.SMOKE)
                if smoke_stats['count'] < 3:
                    p = 100.0 if smoke_stats['count'] == 0 else 80.0
                    candidates.append(self._make_task(spec.name, task, PatientLevel.SMOKE, p))
                    continue

                if not self.CRITERIA[PatientLevel.SMOKE](smoke_stats['best_acc']):
                    if random.random() < 0.01:
                        candidates.append(self._make_task(spec.name, task, PatientLevel.SMOKE, 10.0))
                    continue

                # 2. SHALLOW
                shallow_stats = self._get_stats(progress, spec.name, task, PatientLevel.SHALLOW)
                if shallow_stats['count'] < 10:
                    base_p = 60.0 + (smoke_stats['best_acc'] * 20.0)
                    if shallow_stats['count'] == 0: base_p += 10.0
                    candidates.append(self._make_task(spec.name, task, PatientLevel.SHALLOW, base_p))
                    continue

                if not self.CRITERIA[PatientLevel.SHALLOW](shallow_stats['best_acc']):
                     continue

                # 3. STANDARD (With Verification -> CV)
                std_stats = self._get_stats(progress, spec.name, task, PatientLevel.STANDARD)

                verification_task = self._check_verification_needed(std_stats, spec.name, task, PatientLevel.STANDARD)
                if verification_task:
                    candidates.append(verification_task)

                # Check for Cross-Validation Needs
                cv_task = self._check_cv_needed(std_stats, progress, spec.name, task)
                if cv_task:
                    candidates.append(cv_task)

                if std_stats['count'] < 20:
                    base_p = 40.0 + (shallow_stats['best_acc'] * 30.0)
                    if std_stats['count'] > 15: base_p -= 10.0
                    candidates.append(self._make_task(spec.name, task, PatientLevel.STANDARD, base_p))
                    continue

                if not self.CRITERIA[PatientLevel.STANDARD](std_stats['best_acc']):
                     continue

                # 4. DEEP
                deep_stats = self._get_stats(progress, spec.name, task, PatientLevel.DEEP)

                # Check Robustness (New!)
                robustness_task = self._check_robustness_needed(deep_stats, progress, spec.name, task)
                if robustness_task:
                    candidates.append(robustness_task)

                verification_task = self._check_verification_needed(deep_stats, spec.name, task, PatientLevel.DEEP)
                if verification_task:
                    candidates.append(verification_task)

                if deep_stats['count'] < 5:
                    p = 20.0 + (std_stats['best_acc'] * 50.0)
                    candidates.append(self._make_task(spec.name, task, PatientLevel.DEEP, p))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x.priority + random.uniform(0, 5), reverse=True)
        return candidates[0]

    def _get_stats(self, progress, model, task, tier):
        try:
            return progress[model][task][tier.value]
        except KeyError:
            return {'count': 0, 'best_acc': 0.0, 'trials': []}

    def _check_robustness_needed(self, deep_stats, progress, model, task) -> Optional[ExperimentTask]:
        """
        If a model performs well in DEEP, schedule a robustness check.
        """
        trials = deep_stats.get('trials', [])
        if not trials: return None

        # Check if any deep trial meets the bar
        best_trial = max(trials, key=lambda t: t.accuracy)
        if not self.CRITERIA[PatientLevel.DEEP](best_trial.accuracy):
            return None

        # Check if robustness already run for this model/task
        # We look for a special marker in the DB? Or just assume one check per model/task is enough.
        # Let's assume we store it as a "robustness" tier in our internal tracking
        # But DB stores it as 'deep' tier with 'is_robustness_check' flag?
        # TrialMetrics doesn't expose config flags easily in `get_all_trials` without parsing.

        # Parse all trials to see if any have is_robustness_check=True
        for t in trials:
            if t.config.get('is_robustness_check'):
                return None # Already done

        # Schedule it
        priority = 85.0 + best_trial.accuracy * 10.0
        config_copy = best_trial.config.copy()

        return ExperimentTask(
            model_name=model,
            task_name=task,
            tier=PatientLevel.DEEP,
            study_name=f"{model}_{task}_{PatientLevel.DEEP.value}",
            priority=priority,
            fixed_config=config_copy,
            verification_of_trial_id=best_trial.trial_id,
            is_robustness_check=True
        )

    def _check_cv_needed(self, std_stats, progress, model, task) -> Optional[ExperimentTask]:
        """
        If a model is verified (3+ repeats), check if it has 5-fold CV.
        """
        trials = std_stats.get('trials', [])
        if not trials: return None

        # Find best verified config
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        # Check repeats (is it verified?)
        repeats = 0
        target_config = {k: v for k, v in best_trial.config.items()
                        if k not in ['tier', 'task', 'model', 'epochs', 'batch_size', 'job_id', 'fold']}
        target_hash = hashlib.md5(json.dumps(target_config, sort_keys=True).encode()).hexdigest()

        for t in trials:
             t_conf = {k: v for k, v in t.config.items()
                      if k not in ['tier', 'task', 'model', 'epochs', 'batch_size', 'job_id', 'fold']}
             if hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest() == target_hash:
                 repeats += 1

        if repeats < 3: return None # Not verified yet

        # It is verified. Now check if we have CV trials for this config.
        cv_stats = self._get_stats(progress, model, task, PatientLevel.CROSS_VAL)
        cv_trials = cv_stats.get('trials', [])

        completed_folds = set()
        for t in cv_trials:
            # Check if it matches our target config
            t_conf = {k: v for k, v in t.config.items()
                     if k not in ['tier', 'task', 'model', 'epochs', 'batch_size', 'job_id', 'fold', 'is_verification', 'verified_trial_id']}
            if hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest() == target_hash:
                fold = t.config.get('fold')
                if fold is not None:
                    completed_folds.add(fold)

        # We need folds 0, 1, 2, 3, 4
        for fold in range(5):
            if fold not in completed_folds:
                # Schedule this fold
                config_copy = best_trial.config.copy()

                # Priority: Very High (CV is gold standard)
                priority = 95.0

                return ExperimentTask(
                    model_name=model,
                    task_name=task,
                    tier=PatientLevel.CROSS_VAL,
                    study_name=f"{model}_{task}_{PatientLevel.CROSS_VAL.value}",
                    priority=priority,
                    fixed_config=config_copy,
                    verification_of_trial_id=best_trial.trial_id,
                    fold_index=fold
                )

        return None

    def _check_verification_needed(self, stats, model, task, tier) -> Optional[ExperimentTask]:
        """
        If a trial is very good but hasn't been repeated 3 times, schedule repeats.
        """
        trials = stats.get('trials', [])
        if not trials: return None

        # Sort by accuracy descending
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        if not self.CRITERIA[tier](best_trial.accuracy):
            return None

        repeats = 0
        target_config = {k: v for k, v in best_trial.config.items()
                        if k not in ['tier', 'task', 'model', 'epochs', 'batch_size', 'job_id', 'fold']}

        target_hash = hashlib.md5(json.dumps(target_config, sort_keys=True).encode()).hexdigest()

        for t in trials:
             t_conf = {k: v for k, v in t.config.items()
                      if k not in ['tier', 'task', 'model', 'epochs', 'batch_size', 'job_id', 'fold']}
             if hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest() == target_hash:
                 repeats += 1

        if repeats < 3:
            priority = 90.0 + best_trial.accuracy * 10.0
            config_copy = best_trial.config.copy()
            return ExperimentTask(
                model_name=model,
                task_name=task,
                tier=tier,
                study_name=f"{model}_{task}_{tier.value}",
                priority=priority,
                fixed_config=config_copy,
                verification_of_trial_id=best_trial.trial_id
            )

        return None

    def _make_task(self, model, task, tier, priority):
        return ExperimentTask(
            model_name=model,
            task_name=task,
            tier=tier,
            study_name=f"{model}_{task}_{tier.value}",
            priority=priority
        )


class AutoScientist:
    """
    The main loop.
    """

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self):
        self.state = ExperimentState(DB_PATH)
        self.strategy = ScientistStrategy(self.state)
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
                    logger.critical(f"Too many consecutive failures ({self.consecutive_failures}). Sleeping for 5 minutes.")
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
                if task.tier == PatientLevel.CROSS_VAL: type_str = f"CROSS_VAL (Fold {task.fold_index})"
                elif is_fixed: type_str = "VERIFICATION"
                elif task.is_robustness_check: type_str = "ROBUSTNESS"

                logger.info(f"Starting {type_str}: {task.model_name} | {task.task_name} | {task.tier.name} (Priority: {task.priority:.1f})")

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
                    else:
                        trial = study.ask()
                        config = create_optuna_space(trial, task.model_name)
                        job_id = trial.number

                    # Inject Tier Config
                    tier_config = get_evaluation_config(task.tier)
                    config["epochs"] = tier_config.epochs
                    config["batch_size"] = tier_config.batch_size

                    # Metadata
                    config["tier"] = task.tier.value
                    config["task"] = task.task_name
                    config["model"] = task.model_name
                    if is_fixed:
                        config["is_verification"] = True
                        config["verified_trial_id"] = task.verification_of_trial_id

                    if task.is_robustness_check:
                        config["is_robustness_check"] = True

                    # 3. Execute
                    logger.info(f"  > Config: Epochs={config['epochs']}, Batch={config['batch_size']}")

                    if task.is_robustness_check:
                        # Run Robustness Suite
                        # We use the config but run a special function
                        logger.info("  > Running Robustness Suite...")
                        score = run_robustness_check(task.model_name, task.task_name, config)
                        # We return a dummy metrics dict to store in DB
                        metrics = {
                            "accuracy": score, # Store robustness score as accuracy for now? Or separate field?
                            "loss": 0.0,
                            "robustness_score": score,
                            "time": 0.0,
                            "param_count": 0.0
                        }
                    else:
                        quick = (task.tier == PatientLevel.SMOKE)

                        metrics = run_single_trial_task(
                            task=task.task_name,
                            model_name=task.model_name,
                            config=config,
                            storage_path=DB_PATH,
                            job_id=job_id,
                            quick_mode=quick
                        )

                    # 4. Report
                    if metrics:
                        acc = metrics.get("accuracy", 0.0)
                        loss = metrics.get("loss", float("inf"))
                        logger.info(f"  > Result: Accuracy={acc:.2%}, Loss={loss:.4f}")

                        if trial:
                            study.tell(trial, acc)

                        self.consecutive_failures = 0 # Success!
                    else:
                        logger.warning("  > Trial failed.")
                        if trial:
                            study.tell(trial, 0.0, state=optuna.trial.TrialState.FAIL)

                        self.consecutive_failures += 1

                except Exception as e:
                    logger.error(f"Error executing trial: {e}", exc_info=True)
                    self.consecutive_failures += 1
                    time.sleep(5)

                time.sleep(1)

        finally:
            logger.info("AutoScientist shutting down. Cleaning up...")
            self.state.close()
            logger.info("Shutdown complete.")

if __name__ == "__main__":
    scientist = AutoScientist()
    scientist.run()
