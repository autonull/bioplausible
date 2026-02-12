from typing import Any, Dict, List, Optional
import json

import optuna
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.scientist.failure_tracker import FailureTracker


class ExperimentState:
    """
    Analyzes the current state of research by querying the database.
    Provides aggregated statistics and access to recent experiment history.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.storage = HyperoptStorage(db_path)
        self.failure_tracker = FailureTracker(db_path)

    def get_failure_analysis(self) -> Dict[str, Any]:
        """
        Analyze failure patterns to detect systemic issues.
        """
        return self.failure_tracker.analyze_failure_patterns()

    def get_progress(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns a nested dictionary with stats about completed experiments.

        Structure:
        progress[model_name][task_name][tier_name] = {
            "count": int,
            "best_acc": float,
            "trials": List[Trial],
            "last_run_ts": float
        }
        """
        trials = self.storage.get_all_trials()
        progress: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for t in trials:
            if t.status != "completed":
                continue

            model = t.model_name
            task = t.config.get("task")
            tier_val = t.config.get("tier")

            # Metadata Rescue: Infer Tier from Epochs if missing
            if not tier_val:
                epochs = t.config.get("epochs")
                if epochs:
                    if epochs <= 3:
                        tier_val = "smoke"
                    elif epochs <= 7:
                        tier_val = "shallow"
                    elif epochs <= 15:
                        tier_val = "standard"
                    else:
                        tier_val = "deep"

            if not task or not tier_val:
                continue

            if model not in progress:
                progress[model] = {}
            if task not in progress[model]:
                progress[model][task] = {}
            if tier_val not in progress[model][task]:
                progress[model][task][tier_val] = {
                    "count": 0,
                    "best_acc": -1.0,
                    "trials": [],
                    "last_run_ts": 0.0,
                }

            entry = progress[model][task][tier_val]
            entry["count"] += 1
            entry["trials"].append(t)

            if t.accuracy > entry["best_acc"]:
                entry["best_acc"] = t.accuracy

        return progress

    def get_optuna_study(self, study_name: str) -> optuna.Study:
        """Load or create an Optuna study."""
        return optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{self.db_path}",
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(),
        )

    def get_recent_tasks(self, limit: int = 10) -> List[str]:
        """
        Get list of task names from recently launched trials.

        Args:
            limit: Maximum number of recent tasks to retrieve.

        Returns:
            List of task names.
        """
        try:
            # We need to query hyperopt_logs table via storage
            # Optimization: Use a custom query on the storage connection
            cursor = self.storage.conn.cursor()
            cursor.execute(
                "SELECT config_json FROM hyperopt_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()

            recent_tasks = []
            for row in rows:
                try:
                    config = json.loads(row[0])
                    if "task" in config:
                        recent_tasks.append(config["task"])
                except Exception:
                    pass
            return recent_tasks
        except Exception as e:
            # Fallback
            print(f"Error fetching recent tasks: {e}")
            return []

    def get_fragile_models(
        self, acc_threshold: float = 0.80, robust_threshold: float = 0.40
    ) -> Dict[str, Any]:
        """
        Identify models that have high accuracy but low robustness.

        Args:
            acc_threshold: Minimum accuracy to be considered "performing".
            robust_threshold: Maximum robustness score to be considered "fragile".

        Returns:
            Dict[str, float]: Map of model_name -> avg_robustness_score.
        """
        fragile_models = {}
        try:
            cursor = self.storage.conn.cursor()
            query = """
                SELECT
                    t.model_name,
                    AVG(t.accuracy) as avg_acc,
                    AVG(CASE WHEN ua.key = 'robustness_score' THEN CAST(ua.value_json as REAL) END) as avg_rob
                FROM hyperopt_logs t
                JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
                WHERE t.status = 'completed'
                GROUP BY t.model_name
                HAVING avg_acc > ? AND avg_rob < ? AND avg_rob > 0
            """
            cursor.execute(query, (acc_threshold, robust_threshold))
            rows = cursor.fetchall()

            for row in rows:
                model_name = row["model_name"]
                avg_rob = row["avg_rob"]
                fragile_models[model_name] = avg_rob

        except Exception as e:
            # Table might not exist yet or other DB error
            # print(f"Fragility check failed: {e}")
            pass

        return fragile_models

    def close(self) -> None:
        """Close the database connection."""
        self.storage.close()
