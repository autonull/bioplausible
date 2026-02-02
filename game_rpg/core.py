"""
Core Logic for Research Simulation.
"""

import json
import logging
import time
import copy
import optuna
from typing import Dict, Any, List, Optional
from pathlib import Path

from bioplausible.scientist.core import ExperimentState, ScientistStrategy, ExperimentTask, run_single_trial_task
from bioplausible.hyperopt import PatientLevel, create_optuna_space, get_evaluation_config
from bioplausible.scientist.robustness import run_robustness_check

logger = logging.getLogger("ResearchGame")

class ResearchGame:
    """
    Simulation interface for scientific discovery.
    Focuses on planning, execution, and analysis of experiments.
    """

    # We only store session-specific data here.
    # Long-term data is in the database.
    DEFAULT_SESSION_STATS = {
        "experiments_run_session": 0,
        "session_start": 0.0
    }

    def __init__(self, db_path: str = "bioplausible.db", stats_path: str = "bioplausible_session.json"):
        self.db_path = db_path
        self.stats_path = stats_path
        self.state = ExperimentState(db_path)
        self.strategy = ScientistStrategy(self.state)

        self.stats = self.DEFAULT_SESSION_STATS.copy()
        self.stats["session_start"] = time.time()

        # Load local history (if we want to persist anything UI-related, maybe favorites?)
        # For now, we rely on the DB.

    def get_top_discoveries(self, limit: int = 5) -> List[Any]:
        """
        Query the DB for the best models found so far.
        """
        # We need to query trials from storage.
        # ExperimentState doesn't have a "get_best" method, we can access storage directly.
        try:
            trials = self.state.storage.get_all_trials()
            completed = [t for t in trials if t.status == "completed" and t.accuracy is not None]
            completed.sort(key=lambda x: x.accuracy, reverse=True)
            return completed[:limit]
        except Exception as e:
            logger.error(f"Failed to fetch discoveries: {e}")
            return []

    def get_available_experiments(self) -> List[ExperimentTask]:
        """
        Get candidates from the strategy.
        """
        candidates = self.strategy.generate_candidates()
        # Sort by priority
        candidates.sort(key=lambda x: x.priority, reverse=True)
        return candidates

    def execute_task(self, task: ExperimentTask) -> Optional[float]:
        print(f"\n🔬 Starting Experiment: {task.model_name} on {task.task_name} ({task.tier.name})...")

        try:
            # Prepare Config
            study = self.state.get_optuna_study(task.study_name)
            is_fixed = task.fixed_config is not None

            config = {}
            job_id = None
            trial = None

            if is_fixed:
                config = task.fixed_config
                if task.fold_index is not None:
                    config["fold"] = task.fold_index
            else:
                trial = study.ask()
                config = create_optuna_space(trial, task.model_name)
                job_id = trial.number

            tier_config = get_evaluation_config(task.tier)
            config["epochs"] = tier_config.epochs
            config["batch_size"] = tier_config.batch_size
            config["tier"] = task.tier.value
            config["task"] = task.task_name
            config["model"] = task.model_name
            if is_fixed:
                 config["is_verification"] = True
                 config["verified_trial_id"] = task.verification_of_trial_id
            if task.is_robustness_check:
                 config["is_robustness_check"] = True

            print(f"   ⚙️  Config: Epochs={config['epochs']}, Batch={config['batch_size']}")

            # Simple progress simulation or just let it run
            # run_single_trial_task prints its own logs usually, or we can rely on standard output

            start_time = time.time()
            metrics = None

            if task.is_robustness_check:
                score = run_robustness_check(task.model_name, task.task_name, config)
                metrics = {"accuracy": score, "loss": 0.0}
            else:
                quick = (task.tier == PatientLevel.SMOKE)
                metrics = run_single_trial_task(
                    task=task.task_name,
                    model_name=task.model_name,
                    config=config,
                    storage_path=self.db_path,
                    job_id=job_id,
                    quick_mode=quick
                )

            if metrics:
                acc = metrics.get("accuracy", 0.0)
                loss = metrics.get("loss", 0.0)

                print(f"   ✅ Success! Accuracy: {acc:.2%}, Loss: {loss:.4f}")

                if trial:
                    study.tell(trial, acc)

                self.stats["experiments_run_session"] += 1
                return acc
            else:
                print("   ❌ Experiment Failed.")
                if trial:
                    study.tell(trial, 0.0, state=optuna.trial.TrialState.FAIL)
                return None

        except Exception as e:
            print(f"   💥 Error: {e}")
            logger.error("Experiment failed", exc_info=True)
            return None
