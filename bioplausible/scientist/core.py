"""
AutoScientist: The Autonomous Discovery Agent.

This module implements the core logic for the continuous experiment runner.
It manages the experiment lifecycle:
1. State Analysis: What have we learned so far?
2. Strategy: What should we do next? (Smoke -> Shallow -> Standard -> Deep)
3. Execution: Run the experiment.
4. Learning: Update the knowledge base.
"""

import hashlib
import json
import logging
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import optuna

# psutil needed for resource monitoring
try:
    import psutil
except ImportError:
    psutil = None

from bioplausible.hyperopt import (
    PatientLevel,
    create_constrained_optuna_config,
    get_evaluation_config,
)
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible.scientist.decisions import DecisionLogger
from bioplausible.scientist.robustness import run_robustness_check
from bioplausible.scientist.synthesizer import ResearchSynthesizer
from bioplausible.scientist.curriculum import CurriculumManager
from bioplausible.scientist.promotion import PromotionGate

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scientist.log"), logging.StreamHandler(sys.stdout)],
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
    fixed_config: Optional[Dict[str, Any]] = (
        None  # If set, run this config exactly (verification)
    )
    verification_of_trial_id: Optional[int] = None
    fold_index: Optional[int] = None  # For Cross-Validation (0-4)
    last_run_timestamp: Optional[str] = None
    is_robustness_check: bool = False
    is_ablation: bool = False
    ablation_param: Optional[str] = None
    is_transfer: bool = False
    transfer_from_trial: Optional[int] = None
    is_continual: bool = False
    continual_step: int = 0
    constraints: Optional[Dict[str, Any]] = None  # Constraints for search space


class ResourceMonitor:
    """Monitors system resources to prevent overload."""

    def __init__(self, cpu_limit=90.0, mem_limit=90.0):
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit

    def should_pause(self) -> bool:
        if not psutil:
            return False

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


class ScientistStrategy:
    """
    The Brains. Decides what to run next.
    """

    CRITERIA = {
        PatientLevel.SMOKE: lambda acc: acc > 0.15,
        PatientLevel.SHALLOW: lambda acc: acc > 0.40,
        PatientLevel.STANDARD: lambda acc: acc > 0.60,
        PatientLevel.CROSS_VAL: lambda acc: True,  # CV just needs to run 5 times
        PatientLevel.DEEP: lambda acc: acc > 0.80,  # Deep bar
    }

    def __init__(self, state: ExperimentState, decision_logger: Optional[DecisionLogger] = None):
        self.state = state
        self.decision_logger = decision_logger
        self._logged_events = set()
        self.curriculum = CurriculumManager()

    def _log(self, key, event_type, desc, meta=None):
        if not self.decision_logger:
            return
        if key in self._logged_events:
            return
        self.decision_logger.log_decision(event_type, desc, meta)
        self._logged_events.add(key)

    def generate_candidates(self) -> List[ExperimentTask]:
        """
        Generates a list of all possible valid experiments based on current state.
        """
        progress = self.state.get_progress()
        candidates = []

        # Analyze failures to generate constraints
        failure_constraints = self._analyze_failures(progress)
        if failure_constraints:
            for model, constraints in failure_constraints.items():
                self._log(
                    f"fail_constraint_{model}",
                    "CONSTRAINT_APPLIED",
                    f"High failure rate detected for {model}. Restricting search space.",
                    constraints
                )

        for spec in MODEL_REGISTRY:
            # Map compat names to actual tasks if necessary, or use defaults
            # "vision" -> ["mnist", "cifar10"], "lm" -> ["tiny_shakespeare"], etc.
            tasks = self._resolve_tasks(spec.task_compat, spec.name)

            for task in tasks:
                # 0. CURRICULUM CHECK
                if not self._check_curriculum(progress, spec.name, task):
                    continue

                # 1. SMOKE
                smoke_stats = self._get_stats(
                    progress, spec.name, task, PatientLevel.SMOKE
                )
                if smoke_stats["count"] < 3:
                    if smoke_stats["count"] == 0:
                        self._log(f"smoke_{spec.name}_{task}", "NEW_HYPOTHESIS", f"Starting initial investigation (Smoke Test) for {spec.name} on {task}.")

                    p = 100.0 if smoke_stats["count"] == 0 else 80.0
                    candidates.append(
                        self._make_task(spec.name, task, PatientLevel.SMOKE, p)
                    )
                    continue

                if not self.CRITERIA[PatientLevel.SMOKE](smoke_stats["best_acc"]):
                    if random.random() < 0.01:
                        candidates.append(
                            self._make_task(spec.name, task, PatientLevel.SMOKE, 10.0)
                        )
                    continue

                # 2. SHALLOW
                shallow_stats = self._get_stats(
                    progress, spec.name, task, PatientLevel.SHALLOW
                )
                if shallow_stats["count"] < 10:
                    base_p = 60.0 + (smoke_stats["best_acc"] * 20.0)
                    if shallow_stats["count"] == 0:
                        self._log(f"shallow_{spec.name}_{task}", "PROMOTION", f"Promoting {spec.name} to Shallow Tier (Passed Smoke Test with {smoke_stats['best_acc']:.2%}).")
                        base_p += 10.0

                    # Apply constraints if any
                    model_constraints = failure_constraints.get(spec.name, {})

                    task_obj = self._make_task(spec.name, task, PatientLevel.SHALLOW, base_p)
                    if model_constraints:
                        task_obj.constraints = model_constraints
                    candidates.append(task_obj)
                    continue

                if not self.CRITERIA[PatientLevel.SHALLOW](shallow_stats["best_acc"]):
                    continue

                # 3. STANDARD (With Verification -> CV)
                std_stats = self._get_stats(
                    progress, spec.name, task, PatientLevel.STANDARD
                )

                verification_task = self._check_verification_needed(
                    std_stats, spec.name, task, PatientLevel.STANDARD
                )
                if verification_task:
                    self._log(f"verify_std_{spec.name}_{task}", "VERIFICATION", f"Verifying best result for {spec.name} (Standard Tier).")
                    candidates.append(verification_task)

                # Check for Ablation Studies
                ablation_task = self._check_ablation_needed(
                    std_stats, progress, spec.name, task
                )
                if ablation_task:
                    self._log(f"ablation_{spec.name}_{task}_{ablation_task.ablation_param}", "ABLATION_STUDY", f"Scheduling ablation study for {spec.name} to verify components.", {"param": ablation_task.ablation_param})
                    candidates.append(ablation_task)

                # Check for Continual Learning (Split-MNIST)
                cl_task = self._check_continual_learning_needed(
                    std_stats, progress, spec.name, task
                )
                if cl_task:
                    self._log(f"cl_{spec.name}_{task}_{cl_task.continual_step}", "CONTINUAL_LEARNING", f"Attempting Continual Learning Step {cl_task.continual_step} for {spec.name}.")
                    candidates.append(cl_task)

                # Check for Transfer Learning
                transfer_task = self._check_transfer_needed(
                    std_stats, progress, spec.name, task
                )
                if transfer_task:
                    self._log(f"transfer_{spec.name}_{task}", "TRANSFER_LEARNING", f"Attempting Transfer Learning from {task} for {spec.name}.")
                    candidates.append(transfer_task)

                # Check for Cross-Validation Needs
                cv_task = self._check_cv_needed(std_stats, progress, spec.name, task)
                if cv_task:
                    self._log(f"cv_{spec.name}_{task}", "CROSS_VALIDATION", f"Running 5-Fold Cross-Validation for {spec.name} to confirm stability.")
                    candidates.append(cv_task)

                if std_stats["count"] < 20:
                    base_p = 40.0 + (shallow_stats["best_acc"] * 30.0)
                    if std_stats["count"] == 0:
                         self._log(f"standard_{spec.name}_{task}", "PROMOTION", f"Promoting {spec.name} to Standard Tier (Passed Shallow with {shallow_stats['best_acc']:.2%}).")

                    if std_stats["count"] > 15:
                        base_p -= 10.0

                    # Refine Space based on Shallow results
                    refine_constraints = self._refine_search_space(
                        progress, spec.name, task, PatientLevel.SHALLOW
                    )
                    fail_constraints = failure_constraints.get(spec.name, {})

                    # Merge constraints
                    final_constraints = {}
                    if refine_constraints:
                        self._log(f"refine_std_{spec.name}_{task}", "REFINEMENT", f"Refining search space for Standard Tier based on Shallow results.", refine_constraints)
                        final_constraints.update(refine_constraints)
                    if fail_constraints:
                        final_constraints.update(fail_constraints)

                    task_obj = self._make_task(
                        spec.name, task, PatientLevel.STANDARD, base_p
                    )
                    if final_constraints:
                        task_obj.constraints = final_constraints

                    candidates.append(task_obj)
                    continue

                if not self.CRITERIA[PatientLevel.STANDARD](std_stats["best_acc"]):
                    continue

                # 4. DEEP
                deep_stats = self._get_stats(
                    progress, spec.name, task, PatientLevel.DEEP
                )

                # Check Robustness (New!)
                robustness_task = self._check_robustness_needed(
                    deep_stats, progress, spec.name, task
                )
                if robustness_task:
                    self._log(f"robust_{spec.name}_{task}", "ROBUSTNESS_CHECK", f"Triggering Robustness Analysis for {spec.name} due to high Deep Tier performance.")
                    candidates.append(robustness_task)

                verification_task = self._check_verification_needed(
                    deep_stats, spec.name, task, PatientLevel.DEEP
                )
                if verification_task:
                    candidates.append(verification_task)

                if deep_stats["count"] < 5:
                    if deep_stats["count"] == 0:
                        self._log(f"deep_{spec.name}_{task}", "PROMOTION", f"Promoting {spec.name} to Deep Tier (Passed Standard with {std_stats['best_acc']:.2%}).")

                    p = 20.0 + (std_stats["best_acc"] * 50.0)

                    # Refine Space based on Standard results
                    refine_constraints = self._refine_search_space(
                        progress, spec.name, task, PatientLevel.STANDARD
                    )
                    fail_constraints = failure_constraints.get(spec.name, {})

                    final_constraints = {}
                    if refine_constraints:
                        self._log(f"refine_deep_{spec.name}_{task}", "REFINEMENT", f"Refining search space for Deep Tier based on Standard results.", refine_constraints)
                        final_constraints.update(refine_constraints)
                    if fail_constraints:
                        final_constraints.update(fail_constraints)

                    task_obj = self._make_task(spec.name, task, PatientLevel.DEEP, p)
                    if final_constraints:
                        task_obj.constraints = final_constraints

                    candidates.append(task_obj)

        return candidates

    def _refine_search_space(
        self, progress, model, task, source_tier
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze successful trials from source_tier to refine search space for next tier.
        """
        stats = self._get_stats(progress, model, task, source_tier)
        trials = stats.get("trials", [])

        # Need enough data
        if len(trials) < 3:
            return None

        # Filter for successful trials (top 50%)
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        top_n = max(3, len(trials) // 2)
        top_trials = trials[:top_n]

        # Check if they are actually good (sanity check)
        if top_trials[0].accuracy < 0.2:
            return None

        constraints = {}

        # Analyze LR
        lrs = [t.config["lr"] for t in top_trials if "lr" in t.config]
        if lrs:
            min_lr = min(lrs)
            max_lr = max(lrs)
            # Relax bounds: 0.5x to 2.0x, but centered
            constraints["min_lr"] = min_lr * 0.5
            constraints["max_lr"] = max_lr * 2.0

        # Analyze Beta
        betas = [
            t.config["beta"]
            for t in top_trials
            if "beta" in t.config and t.config["beta"] is not None
        ]
        if betas:
            min_beta = min(betas)
            max_beta = max(betas)
            constraints["min_beta"] = max(0.0, min_beta - 0.1)
            constraints["max_beta"] = min(1.0, max_beta + 0.1)

        return constraints

    def _analyze_failures(self, progress) -> Dict[str, Dict[str, Any]]:
        """
        Analyze failure rates to suggest constraints.
        Returns: Dict[model_name, constraint_dict]
        """
        constraints = {}
        for model, task_data in progress.items():
            # Aggregate stats across all tasks/tiers for this model
            total = 0
            failures = 0
            for task, tier_data in task_data.items():
                for tier, stats in tier_data.items():
                    # We need to count failed trials.
                    # Progress dict only has 'completed'.
                    # We need to look at raw trials in DB for full stats,
                    # but 'progress' is pre-filtered.
                    # For now, we'll use a heuristic: if we have many trials but low best_acc,
                    # or we can try to infer stability issues if 'smoke' tier has low success rate.

                    # Better approach: check recent trials in the trials list
                    trials = stats.get("trials", [])
                    for t in trials:
                        total += 1
                        if t.final_loss > 100 or t.accuracy < 0.11: # Divergence or random chance
                            failures += 1

            if total > 5 and (failures / total) > 0.3:
                # High failure rate -> Constrain search space
                # Heuristic: Reduce LR and Beta
                constraints[model] = {
                    "max_lr": 0.005,
                    "max_beta": 0.5
                }
        return constraints

    def plan_next(self) -> Optional[ExperimentTask]:
        """
        Scans all possibilities and returns the highest priority experiment.
        """
        candidates = self.generate_candidates()

        if not candidates:
            return None

        candidates.sort(key=lambda x: x.priority + random.uniform(0, 5), reverse=True)
        return candidates[0]

    def _resolve_tasks(self, task_compat: List[str], model_name: str = "") -> List[str]:
        """
        Convert compatibility list to specific runnable tasks.
        Uses CurriculumManager to refine choices.
        """
        if not task_compat:
            # If no specific compat, ask curriculum for starting point
            initial = self.curriculum.get_initial_task(model_name)
            return [initial] if initial else ["mnist"]

        # ... (rest of logic can be simplified or kept if needed for specific overrides)
        # For now, let's trust the compat list but filter by validity if we had a full registry
        resolved = []
        for t in task_compat:
            if t == "vision":
                resolved.extend(["mnist", "fashion_mnist", "cifar10"])
            elif t == "lm":
                resolved.extend(["char_ngram", "tiny_shakespeare"])
            elif t == "rl":
                    resolved.extend(["cartpole", "pendulum"])
            else:
                resolved.append(t)
        return list(set(resolved))

    def _check_curriculum(self, progress: Dict, model_name: str, task: str) -> bool:
        """
        Check if we are allowed to run this task based on curriculum.
        """
        # If task is an initial task, allow it (unless we want to enforce sequential completion of tracks)
        # Ideally we check what track this task belongs to and see if previous tasks are done.
        
        # Simplified logic using CurriculumManager tracks manually for now without full state tracking in Manager:
        # We need to find the prerequisite for 'task'
        
        # Find track and index
        track = None
        for t_list in self.curriculum.TRACKS.values():
            if task in t_list:
                track = t_list
                break
        
        if not track:
            return True # Unknown task, assume independent
            
        try:
            curr_idx = track.index(task)
        except ValueError:
            return True

        if curr_idx == 0:
            return True # First task in track is always allowed
            
        prev_task = track[curr_idx - 1]
        
        # Check if prev_task is "passed" for this model
        # We define "passed" as meeting the promotion threshold
        
        # Get best stats for prev_task
        # We need to look across all tiers. Usually 'standard' or 'shallow' is enough.
        # Let's check the highest tier attempted or just raw metrics max.
        
        # We need to query the progress dict structure: progress[model][task][tier] -> {best_acc, count}
        if model_name not in progress or prev_task not in progress[model_name]:
            return False # Prereq not started
            
        # Aggregate best metrics across tiers
        best_metrics = {"accuracy": 0.0, "reward": -float('inf')}
        tiers_run = False
        
        for tier_data in progress[model_name][prev_task].values():
            if tier_data.get("count", 0) > 0:
                tiers_run = True
                if "best_acc" in tier_data:
                    best_metrics["accuracy"] = max(best_metrics["accuracy"], tier_data["best_acc"])
                # We need to handle reward if we tracked it in progress dict (currently progress might only have accuracy?)
                # Assuming progress dict structure from ExperimentState
        
        if not tiers_run:
            return False
            
        # Check promotion
        if PromotionGate.check_promotion(prev_task, best_metrics):
             return True
        else:
             # Log once why blocked?
             return False

    def _get_stats(self, progress, model, task, tier):
        try:
            return progress[model][task][tier.value]
        except KeyError:
            return {"count": 0, "best_acc": 0.0, "trials": []}

    def _check_continual_learning_needed(
        self, stats, progress, model, task
    ) -> Optional[ExperimentTask]:
        """
        Schedule next steps in a Split-MNIST Continual Learning sequence.
        Sequence: mnist_01 -> mnist_23 -> mnist_45 -> mnist_67 -> mnist_89
        """
        if task != "mnist":
            return None

        # Check if base MNIST is mastered
        if stats["count"] == 0 or stats["best_acc"] < 0.95:
            return None

        # Sequence definition
        steps = [
            ("mnist_01", 0),
            ("mnist_23", 1),
            ("mnist_45", 2),
            ("mnist_67", 3),
            ("mnist_89", 4),
        ]

        previous_trial_id = None

        for i, (step_task, step_idx) in enumerate(steps):
            step_stats = self._get_stats(
                progress, model, step_task, PatientLevel.STANDARD
            )

            if step_stats["count"] == 0:
                # Need to run this step
                config_copy = {}

                if step_idx > 0:
                    if previous_trial_id is None:
                        return None  # Cannot proceed

                    # Get config from previous best
                    prev_task_name = steps[i - 1][0]
                    prev_stats = self._get_stats(
                        progress, model, prev_task_name, PatientLevel.STANDARD
                    )
                    best_prev = max(prev_stats["trials"], key=lambda t: t.accuracy)
                    config_copy = best_prev.config.copy()
                    config_copy["transfer_from"] = best_prev.trial_id
                    config_copy["freeze_layers"] = False
                else:
                    # Step 0: Use best MNIST config
                    best_mnist = max(stats["trials"], key=lambda t: t.accuracy)
                    config_copy = best_mnist.config.copy()

                config_copy["is_continual"] = True
                config_copy["continual_step"] = step_idx

                return ExperimentTask(
                    model_name=model,
                    task_name=step_task,
                    tier=PatientLevel.STANDARD,
                    study_name=f"{model}_mnist_cl_step{step_idx}",
                    priority=98.0 + (step_idx * 0.1),
                    fixed_config=config_copy,
                    is_continual=True,
                    continual_step=step_idx,
                    transfer_from_trial=config_copy.get("transfer_from"),
                )

            # Check if this step was successful
            best_step_trial = max(step_stats["trials"], key=lambda t: t.accuracy)
            if best_step_trial.accuracy < 0.80:
                return None  # Failed

            previous_trial_id = best_step_trial.trial_id

        return None

    def _check_transfer_needed(
        self, stats, progress, model, task
    ) -> Optional[ExperimentTask]:
        """
        If a model masters a base task (e.g. MNIST), try transferring to a related harder task (Fashion).
        """
        # Only transfer FROM mnist
        if task != "mnist":
            return None

        trials = stats.get("trials", [])
        if not trials:
            return None

        # Check performance
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        if best_trial.accuracy < 0.90:
            return None  # Not good enough to transfer

        # Target: Fashion MNIST
        target_task = "fashion_mnist"

        # Check if already attempted transfer
        # We look in the target task stats for this model
        target_stats = self._get_stats(
            progress, model, target_task, PatientLevel.STANDARD
        )

        # Heuristic: If we haven't tried ANY standard FashionMNIST runs for this model,
        # or we haven't tried THIS specific transfer.
        # Let's just check if we have done *transfer* specifically.
        already_done = False
        for t in target_stats.get("trials", []):
            if t.config.get("transfer_from") == best_trial.trial_id:
                already_done = True
                break

        if not already_done:
            config_copy = best_trial.config.copy()
            config_copy["transfer_from"] = best_trial.trial_id
            config_copy["freeze_layers"] = True  # Test feature reuse

            return ExperimentTask(
                model_name=model,
                task_name=target_task,
                tier=PatientLevel.STANDARD,
                study_name=f"{model}_{target_task}_transfer",
                priority=92.0,  # High priority
                fixed_config=config_copy,
                is_transfer=True,
                transfer_from_trial=best_trial.trial_id,
            )

        return None

    def _check_ablation_needed(
        self, stats, progress, model, task
    ) -> Optional[ExperimentTask]:
        """
        If a model performs well, schedule ablation studies to understand why.
        """
        trials = stats.get("trials", [])
        if not trials:
            return None

        # Find best trial
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        # Only ablate if it meets the bar
        if not self.CRITERIA[PatientLevel.STANDARD](best_trial.accuracy):
            return None

        # Determine possible ablations based on config
        # This is a simple heuristic list
        ablations = []
        config = best_trial.config

        # 1. Symmetric Weights (if explicit)
        if "symmetric_weights" in config:
            ablations.append(("symmetric_weights", not config["symmetric_weights"]))

        # 2. Feedback Alignment (if explicit or inferred)
        # If 'beta' > 0 (EqProp), maybe try beta=0?
        if config.get("beta", 0.0) > 0.0:
            ablations.append(("beta", 0.0))

        # 3. Top-Down Feedback
        if config.get("use_top_down", False):
            ablations.append(("use_top_down", False))

        for param, val in ablations:
            # Check if this ablation has already been run
            already_run = False
            for t in trials:
                if (
                    t.config.get("is_ablation")
                    and t.config.get("ablation_param") == param
                ):
                    already_run = True
                    break

            if not already_run:
                # Schedule it
                config_copy = config.copy()
                config_copy[param] = val
                config_copy["is_ablation"] = True
                config_copy["ablation_param"] = param

                # Priority: Moderate-High
                priority = 80.0

                return ExperimentTask(
                    model_name=model,
                    task_name=task,
                    tier=PatientLevel.STANDARD,  # Run at standard tier
                    study_name=f"{model}_{task}_{PatientLevel.STANDARD.value}",
                    priority=priority,
                    fixed_config=config_copy,
                    verification_of_trial_id=best_trial.trial_id,
                    is_ablation=True,
                    ablation_param=param,
                )

        return None

    def _check_robustness_needed(
        self, deep_stats, progress, model, task
    ) -> Optional[ExperimentTask]:
        """
        If a model performs well in DEEP, schedule a robustness check.
        """
        trials = deep_stats.get("trials", [])
        if not trials:
            return None

        # Check if any deep trial meets the bar
        best_trial = max(trials, key=lambda t: t.accuracy)
        if not self.CRITERIA[PatientLevel.DEEP](best_trial.accuracy):
            return None

        # Check if robustness already run for this model/task.
        # We look for a special marker 'is_robustness_check' in the DB config.
        # TrialMetrics doesn't expose config flags easily in `get_all_trials` without parsing.

        # Parse all trials to see if any have is_robustness_check=True
        for t in trials:
            if t.config.get("is_robustness_check"):
                return None  # Already done

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
            is_robustness_check=True,
        )

    def _check_cv_needed(
        self, std_stats, progress, model, task
    ) -> Optional[ExperimentTask]:
        """
        If a model is verified (3+ repeats), check if it has 5-fold CV.
        """
        trials = std_stats.get("trials", [])
        if not trials:
            return None

        # Find best verified config
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        # Check repeats (is it verified?)
        repeats = 0
        target_config = {
            k: v
            for k, v in best_trial.config.items()
            if k
            not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
        }
        target_hash = hashlib.md5(
            json.dumps(target_config, sort_keys=True).encode()
        ).hexdigest()

        for t in trials:
            t_conf = {
                k: v
                for k, v in t.config.items()
                if k
                not in [
                    "tier",
                    "task",
                    "model",
                    "epochs",
                    "batch_size",
                    "job_id",
                    "fold",
                ]
            }
            if (
                hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest()
                == target_hash
            ):
                repeats += 1

        if repeats < 3:
            return None  # Not verified yet

        # It is verified. Now check if we have CV trials for this config.
        cv_stats = self._get_stats(progress, model, task, PatientLevel.CROSS_VAL)
        cv_trials = cv_stats.get("trials", [])

        completed_folds = set()
        for t in cv_trials:
            # Check if it matches our target config
            t_conf = {
                k: v
                for k, v in t.config.items()
                if k
                not in [
                    "tier",
                    "task",
                    "model",
                    "epochs",
                    "batch_size",
                    "job_id",
                    "fold",
                    "is_verification",
                    "verified_trial_id",
                ]
            }
            if (
                hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest()
                == target_hash
            ):
                fold = t.config.get("fold")
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
                    fold_index=fold,
                )

        return None

    def _check_verification_needed(
        self, stats, model, task, tier
    ) -> Optional[ExperimentTask]:
        """
        If a trial is very good but hasn't been repeated 3 times, schedule repeats.
        """
        trials = stats.get("trials", [])
        if not trials:
            return None

        # Sort by accuracy descending
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        if not self.CRITERIA[tier](best_trial.accuracy):
            return None

        repeats = 0
        target_config = {
            k: v
            for k, v in best_trial.config.items()
            if k
            not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
        }

        target_hash = hashlib.md5(
            json.dumps(target_config, sort_keys=True).encode()
        ).hexdigest()

        for t in trials:
            t_conf = {
                k: v
                for k, v in t.config.items()
                if k
                not in [
                    "tier",
                    "task",
                    "model",
                    "epochs",
                    "batch_size",
                    "job_id",
                    "fold",
                ]
            }
            if (
                hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest()
                == target_hash
            ):
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
                verification_of_trial_id=best_trial.trial_id,
            )

        return None

    def _make_task(self, model, task, tier, priority):
        return ExperimentTask(
            model_name=model,
            task_name=task,
            tier=tier,
            study_name=f"{model}_{task}_{tier.value}",
            priority=priority,
        )


class AutoScientist:
    """
    The main loop.
    """

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.state = ExperimentState(db_path)
        self.decision_logger = DecisionLogger(db_path)
        self.strategy = ScientistStrategy(self.state, self.decision_logger)
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
                        else:
                            job_id = f"Fixed-{task.study_name}"

                    else:
                        trial = study.ask()
                        # Pass dynamic constraints (intelligence)
                        constraints = {}
                        if task.constraints:
                            constraints.update(task.constraints)
                            logger.info(f"  > Applying intelligent constraints: {constraints}")

                        config = create_constrained_optuna_config(
                            trial, 
                            task.model_name, 
                            custom_constraints=constraints
                        )
                        job_id = trial.number if trial.number is not None else "Unknown"
                        
                        # Log metadata for reports
                        trial.set_user_attr("model_name", task.model_name)
                        trial.set_user_attr("task_name", task.task_name)
                        trial.set_user_attr("tier", task.tier.value)

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
            logger.info("✓ Modular report generated (01_summary.md, 03_leaderboards.md, FULL_REPORT.md)")
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
                        f.write(f"| {i} | {r['model']} | {r['best_accuracy']:.2%} | {r['mean_accuracy']:.2%} | {r['std']:.4f} | {r['trials']} |\n")
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
                            f.write(f"{i}. **{w['model']}**: {w['accuracy']:.2%} ({w['params']:,} params)\n")
                        f.write("\n")
                
                # Efficiency Analysis  
                f.write("## ⚡ Efficiency Analysis\n\n")
                efficiency = synthesis_result.get("efficiency_analysis", {})
                
                if "top_epoch_efficient" in efficiency:
                    f.write("### Top Models by Epoch Efficiency (Accuracy / Epoch)\n")
                    f.write("*Models that converge fastest - high accuracy with fewer epochs.*\n\n")
                    f.write("| Model | Task | Accuracy | Epochs | Acc/Epoch |\n")
                    f.write("|-------|------|----------|--------|----------|\n")
                    for r in efficiency["top_epoch_efficient"][:5]:
                        eff = r['epoch_efficiency']
                        f.write(f"| {r['model_name']} | {r['task_name']} | {r['accuracy']:.2%} | {r['num_epochs']} | {eff:.4f} |\n")
                    f.write("\n")
                
                if "top_param_efficient" in efficiency:
                    f.write("### Top Models by Parameter Efficiency (Accuracy / M-Params)\n")
                    f.write("*Models that achieve high performance with fewer parameters.*\n\n")
                    for r in efficiency["top_param_efficient"][:5]:
                        params_m = r['param_count'] / 1e6
                        f.write(f"- **{r['model_name']}**: {r['accuracy']:.2%} with {params_m:.2f}M params (efficiency: {r['param_efficiency']:.2f})\n")
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
