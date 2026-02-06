from typing import Any, Dict
import optuna
from bioplausible.hyperopt.storage import HyperoptStorage

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

    def get_optuna_study(self, study_name: str):
        """Load or create an Optuna study."""
        return optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{self.db_path}",
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(),
        )

    def close(self):
        self.storage.close()
