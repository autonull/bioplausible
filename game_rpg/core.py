"""
Core Logic for Research RPG.
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

from .upgrades import UpgradeManager
from .quests import QuestManager

logger = logging.getLogger("ResearchGame")

class ResearchGame:
    """
    Gamified interface for scientific discovery.
    """

    DEFAULT_STATS = {
        "level": 1,
        "xp": 0,
        "science_points": 0.0,
        "experiments_run": 0,
        "upgrades": {}, # id -> level
        "quests": [], # list of active quest dicts
        "discoveries": []
    }

    LEVEL_THRESHOLDS = {
        1: 0,
        2: 100,
        3: 300,
        4: 600,
        5: 1000,
        6: 1500,
        7: 2100,
        8: 2800,
        9: 3600,
        10: 4500
    }

    def __init__(self, db_path: str = "bioplausible.db", stats_path: str = "bioplausible_gamestate.json"):
        self.db_path = db_path
        self.stats_path = stats_path
        self.state = ExperimentState(db_path)
        self.strategy = ScientistStrategy(self.state)

        self.stats = self.load_stats()

        # Sub-Managers
        self.upgrade_manager = UpgradeManager(self)
        self.quest_manager = QuestManager(self)

        # Ensure daily quests are populated on load
        self.quest_manager.refresh_daily_quests()

    def load_stats(self) -> Dict[str, Any]:
        path = Path(self.stats_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")
                return copy.deepcopy(self.DEFAULT_STATS)
        return copy.deepcopy(self.DEFAULT_STATS)

    def save_stats(self):
        try:
            with open(self.stats_path, "w") as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    @property
    def level(self) -> int:
        return self.stats.get("level", 1)

    @property
    def xp(self) -> int:
        return self.stats.get("xp", 0)

    @property
    def science_points(self) -> float:
        return self.stats.get("science_points", 0.0)

    def add_xp(self, amount: int):
        # Apply Multipliers
        mult = self.upgrade_manager.get_multiplier("xp_gain")
        final_amount = int(amount * mult)

        self.stats["xp"] = self.xp + final_amount
        self._check_level_up()
        self.save_stats()

    def add_science_points(self, amount: float):
        # Apply Multipliers
        mult = self.upgrade_manager.get_multiplier("sp_gain")
        final_amount = amount * mult

        self.stats["science_points"] = self.science_points + final_amount
        self.save_stats()

    def _check_level_up(self):
        current_level = self.level
        xp = self.xp

        new_level = current_level
        for lvl, threshold in sorted(self.LEVEL_THRESHOLDS.items()):
            if xp >= threshold:
                new_level = max(new_level, lvl)

        if new_level > current_level:
            self.stats["level"] = new_level
            print(f"\n🎉 LEVEL UP! You are now Level {new_level}! 🎉\n")
            # Refresh quests on level up?
            self.quest_manager.generate_quests()

    def get_available_experiments(self) -> List[ExperimentTask]:
        candidates = self.strategy.generate_candidates()
        available = []

        for task in candidates:
            # Level Restrictions
            required_level = 1
            if task.tier == PatientLevel.STANDARD:
                required_level = 2
            elif task.tier == PatientLevel.CROSS_VAL:
                required_level = 5
            elif task.tier == PatientLevel.DEEP:
                required_level = 8
            elif task.is_robustness_check:
                required_level = 10

            if self.level >= required_level:
                available.append(task)

        available.sort(key=lambda x: x.priority, reverse=True)
        return available

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

                # Base Rewards
                xp_gain = 10
                if task.tier == PatientLevel.SHALLOW: xp_gain = 25
                elif task.tier == PatientLevel.STANDARD: xp_gain = 100
                elif task.tier == PatientLevel.CROSS_VAL: xp_gain = 200
                elif task.tier == PatientLevel.DEEP: xp_gain = 500

                sp_gain = acc * 10.0
                if task.tier == PatientLevel.STANDARD: sp_gain *= 5
                if task.tier == PatientLevel.DEEP: sp_gain *= 20

                print(f"   🏆 Rewards: +{xp_gain} XP (Base), +{sp_gain:.2f} SP (Base)")
                self.add_xp(xp_gain)
                self.add_science_points(sp_gain)

                self.stats["experiments_run"] = self.stats.get("experiments_run", 0) + 1

                # Check Quests
                self.quest_manager.check_quests(task, metrics)

                self.save_stats()
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
