import hashlib
import json
import logging
import random
from typing import Any, Dict, List, Optional

from bioplausible.hyperopt import PatientLevel
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible.scientist.curriculum import CurriculumManager
from bioplausible.scientist.decisions import DecisionLogger
from bioplausible.scientist.promotion import PromotionGate
from bioplausible.scientist.state import ExperimentState
from bioplausible.scientist.task import ExperimentTask

logger = logging.getLogger("AutoScientist")


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

    TASK_WEIGHTS = {
        "char_ngram": 0.35,
        "tiny_shakespeare": 0.25,
        "pendulum": 0.25,
        "cartpole": 0.05,
        "cifar10": 0.05,
        "cifar100": 0.02,
        "mnist": 0.02,
        "fashion_mnist": 0.01
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

        # Analyze saturation (Tasks that are "solved")
        saturated_tasks = self._analyze_saturation(progress)
        if saturated_tasks:
            for model, tasks in saturated_tasks.items():
                for t in tasks:
                    self._log(f"saturation_{model}_{t}", "SATURATION",
                              f"Task {t} is saturated (solved) for {model}. Skipping.")

        for spec in MODEL_REGISTRY:
            # Map compat names to actual tasks if necessary, or use defaults
            # "vision" -> ["mnist", "cifar10"], "lm" -> ["tiny_shakespeare"], etc.
            tasks = self._resolve_tasks(spec.task_compat, spec.name)

            for task in tasks:
                # Check saturation
                if spec.name in saturated_tasks and task in saturated_tasks[spec.name]:
                    continue

                # 0. CURRICULUM CHECK
                if not self._check_curriculum(progress, spec.name, task):
                    continue

                # 1. SMOKE
                smoke_stats = self._get_stats(
                    progress, spec.name, task, PatientLevel.SMOKE
                )
                if smoke_stats["count"] < 3:
                    if smoke_stats["count"] == 0:
                        self._log(f"smoke_{spec.name}_{task}", "NEW_HYPOTHESIS",
                                  f"Starting initial investigation (Smoke Test) for {spec.name} on {task}.")

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
                    # Tuned Priority: Slightly reduced to balance breadth vs depth
                    # Old: 60 + acc*20 (Max 80)
                    # New: 50 + acc*30 (Max 80) but starts lower
                    base_p = 50.0 + (smoke_stats["best_acc"] * 30.0)
                    if shallow_stats["count"] == 0:
                        self._log(f"shallow_{spec.name}_{task}", "PROMOTION",
                                  f"Promoting {spec.name} to Shallow Tier (Passed Smoke Test with {smoke_stats['best_acc']:.2%}).")
                        base_p += 10.0

                    # Apply constraints if any
                    model_constraints = failure_constraints.get(spec.name, {})

                    task_obj = self._make_task(
                        spec.name, task, PatientLevel.SHALLOW, base_p)
                    if model_constraints:
                        task_obj.constraints = model_constraints
                    candidates.append(task_obj)
                    continue

                if not self.CRITERIA[PatientLevel.SHALLOW](shallow_stats["best_acc"]):
                    self._log(f"stagnated_shallow_{spec.name}_{task}", "STAGNATION", f"Model {spec.name} failed Shallow Tier on {task} (Acc: {shallow_stats['best_acc']:.2%}). Stopping.", {
                              "best_acc": shallow_stats["best_acc"]})
                    continue

                # 3. STANDARD (With Verification -> CV)
                std_stats = self._get_stats(
                    progress, spec.name, task, PatientLevel.STANDARD
                )

                verification_task = self._check_verification_needed(
                    std_stats, spec.name, task, PatientLevel.STANDARD
                )
                if verification_task:
                    self._log(f"verify_std_{spec.name}_{task}", "VERIFICATION",
                              f"Verifying best result for {spec.name} (Standard Tier).")
                    candidates.append(verification_task)

                # Check for Low-Data Regime (Phase 6.1)
                low_data_task = self._check_low_data_needed(
                    std_stats, progress, spec.name, task
                )
                if low_data_task:
                    self._log(f"low_data_{spec.name}_{task}", "LOW_DATA_REGIME",
                              f"Scheduling Low-Data experiment ({low_data_task.fixed_config['data_fraction']:.0%}) for {spec.name}.")
                    candidates.append(low_data_task)

                # Check for Ablation Studies
                ablation_task = self._check_ablation_needed(
                    std_stats, progress, spec.name, task
                )
                if ablation_task:
                    self._log(f"ablation_{spec.name}_{task}_{ablation_task.ablation_param}", "ABLATION_STUDY",
                              f"Scheduling ablation study for {spec.name} to verify components.", {"param": ablation_task.ablation_param})
                    candidates.append(ablation_task)

                # Check for Continual Learning (Split-MNIST)
                cl_task = self._check_continual_learning_needed(
                    std_stats, progress, spec.name, task
                )
                if cl_task:
                    self._log(f"cl_{spec.name}_{task}_{cl_task.continual_step}", "CONTINUAL_LEARNING",
                              f"Attempting Continual Learning Step {cl_task.continual_step} for {spec.name}.")
                    candidates.append(cl_task)

                # Check for Transfer Learning
                transfer_task = self._check_transfer_needed(
                    std_stats, progress, spec.name, task
                )
                if transfer_task:
                    self._log(f"transfer_{spec.name}_{task}", "TRANSFER_LEARNING",
                              f"Attempting Transfer Learning from {task} for {spec.name}.")
                    candidates.append(transfer_task)

                # Check for Cross-Validation Needs
                cv_task = self._check_cv_needed(std_stats, progress, spec.name, task)
                if cv_task:
                    self._log(f"cv_{spec.name}_{task}", "CROSS_VALIDATION",
                              f"Running 5-Fold Cross-Validation for {spec.name} to confirm stability.")
                    candidates.append(cv_task)

                if std_stats["count"] < 20:
                    # Tuned Priority: Increased base and multiplier to encourage depth
                    # Old: 40 + acc*30 (Max 70)
                    # New: 60 + acc*40 (Max 100)
                    base_p = 60.0 + (shallow_stats["best_acc"] * 40.0)

                    if std_stats["count"] == 0:
                        self._log(f"standard_{spec.name}_{task}", "PROMOTION",
                                  f"Promoting {spec.name} to Standard Tier (Passed Shallow with {shallow_stats['best_acc']:.2%}).")

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
                        self._log(f"refine_std_{spec.name}_{task}", "REFINEMENT",
                                  f"Refining search space for Standard Tier based on Shallow results.", refine_constraints)
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
                    self._log(f"robust_{spec.name}_{task}", "ROBUSTNESS_CHECK",
                              f"Triggering Robustness Analysis for {spec.name} due to high Deep Tier performance.")
                    candidates.append(robustness_task)

                verification_task = self._check_verification_needed(
                    deep_stats, spec.name, task, PatientLevel.DEEP
                )
                if verification_task:
                    candidates.append(verification_task)

                if deep_stats["count"] < 5:
                    if deep_stats["count"] == 0:
                        self._log(f"deep_{spec.name}_{task}", "PROMOTION",
                                  f"Promoting {spec.name} to Deep Tier (Passed Standard with {std_stats['best_acc']:.2%}).")

                    p = 20.0 + (std_stats["best_acc"] * 50.0)

                    # Refine Space based on Standard results
                    refine_constraints = self._refine_search_space(
                        progress, spec.name, task, PatientLevel.STANDARD
                    )
                    fail_constraints = failure_constraints.get(spec.name, {})

                    final_constraints = {}
                    if refine_constraints:
                        self._log(f"refine_deep_{spec.name}_{task}", "REFINEMENT",
                                  f"Refining search space for Deep Tier based on Standard results.", refine_constraints)
                        final_constraints.update(refine_constraints)
                    if fail_constraints:
                        final_constraints.update(fail_constraints)

                    task_obj = self._make_task(spec.name, task, PatientLevel.DEEP, p)
                    if final_constraints:
                        task_obj.constraints = final_constraints

                    candidates.append(task_obj)

        # Apply Task Rebalancing (Phase 2.1) & Future Value Boost (Bias Fix)
        for c in candidates:
            weight = self.TASK_WEIGHTS.get(c.task_name, 0.10)  # Default 0.10

            # Future Value Boost: Check if this task unlocks a higher value task (Deep Lookahead)
            future_boost = 0.0
            for track_name, track_tasks in self.curriculum.TRACKS.items():
                if c.task_name in track_tasks:
                    idx = track_tasks.index(c.task_name)
                    # Look ahead at all future tasks in the track
                    for forward_idx in range(idx + 1, len(track_tasks)):
                        future_task = track_tasks[forward_idx]
                        future_weight = self.TASK_WEIGHTS.get(future_task, 0.10)

                        if future_weight > weight:
                            distance = forward_idx - idx
                            # Discount future value by distance (0.9 decay)
                            # e.g. dist=1 -> 0.9, dist=2 -> 0.81
                            boost = (future_weight - weight) * (0.9 ** distance)
                            if boost > future_boost:
                                future_boost = boost

            # Effective weight combines current value and future potential
            effective_weight = weight + future_boost

            # Normalize impact: 0.20 weight -> 1.0 multiplier (Neutral)
            c.priority *= (effective_weight * 5.0)

        # Apply Diversity Penalty (Variety)
        recent_tasks = self.state.get_recent_tasks(limit=10)
        task_counts = {}
        for t in recent_tasks:
            task_counts[t] = task_counts.get(t, 0) + 1

        for c in candidates:
            # Check how many times this task appears in recent history
            count = task_counts.get(c.task_name, 0)
            if count > 0:
                # Decay priority: 0.9^count
                # 1 run: 0.9x
                # 3 runs: 0.729x
                # 5 runs: 0.59x
                penalty = 0.9 ** count
                c.priority *= penalty

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
                        if t.final_loss > 100 or t.accuracy < 0.11:  # Divergence or random chance
                            failures += 1

            if total > 5 and (failures / total) > 0.3:
                # High failure rate -> Constrain search space
                # Heuristic: Reduce LR and Beta
                constraints[model] = {
                    "max_lr": 0.005,
                    "max_beta": 0.5
                }
        return constraints

    def _analyze_saturation(self, progress) -> Dict[str, List[str]]:
        """
        Identify tasks that are effectively "solved" (saturated) for a given model.
        Returns: Dict[model, List[task_name]]
        """
        saturated = {}
        for model, task_data in progress.items():
            for task, tier_data in task_data.items():
                solved_count = 0
                for tier, stats in tier_data.items():
                    trials = stats.get("trials", [])
                    for t in trials:
                        # Check for saturation (e.g. > 99.5% accuracy)
                        if t.accuracy > 0.995:
                            solved_count += 1

                if solved_count >= 5:
                # EFFICIENT SATURATION CHECK (User Requested)
                # Only mark as saturated if we have found an EFFICIENT solution.
                # If we only have large models solving it, keep searching for smaller ones.
                    solved_trials = [t for t in trials if t.accuracy > 0.995]
                    min_params = float('inf')
                    if solved_trials:
                        # param_count is in millions
                        params = [t.param_count for t in solved_trials if t.param_count is not None]
                        if params:
                            min_params = min(params)
                    
                    # Threshold: 50k parameters (0.05M)
                    # If smallest successful model is > 50k params, NOT saturated.
                    if min_params > 0.05:
                        continue

                    if model not in saturated:
                        saturated[model] = []
                    saturated[model].append(task)
        return saturated

    def plan_next(self) -> Optional[ExperimentTask]:
        """
        Scans all possibilities and returns the highest priority experiment.
        """
        candidates = self.generate_candidates()

        if not candidates:
            return None

        # Standard Tier Calibration (Force 50 Standard Trials)
        # Check global standard trials count
        progress = self.state.get_progress()
        total_standard_trials = 0
        for model in progress.values():
            for task in model.values():
                if PatientLevel.STANDARD.value in task:
                    total_standard_trials += task[PatientLevel.STANDARD.value]["count"]

        if total_standard_trials < 50:
            boost_applied = False
            for c in candidates:
                if c.tier == PatientLevel.STANDARD:
                    c.priority += 500.0  # Massive boost
                    boost_applied = True

            if boost_applied:
                logger.info(
                    f"Calibration Mode Active: Boosted Standard Tier candidates (Count: {total_standard_trials}/50)")

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
                resolved.extend(["mnist", "fashion_mnist", "cifar10", "cifar100"])
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
            return True  # Unknown task, assume independent

        try:
            curr_idx = track.index(task)
        except ValueError:
            return True

        if curr_idx == 0:
            return True  # First task in track is always allowed

        prev_task = track[curr_idx - 1]

        # Check if prev_task is "passed" for this model
        # We define "passed" as meeting the promotion threshold

        # Get best stats for prev_task
        # We need to look across all tiers. Usually 'standard' or 'shallow' is enough.
        # Let's check the highest tier attempted or just raw metrics max.

        # We need to query the progress dict structure: progress[model][task][tier] -> {best_acc, count}
        if model_name not in progress or prev_task not in progress[model_name]:
            return False  # Prereq not started

        # Aggregate best metrics across tiers
        best_metrics = {"accuracy": 0.0, "reward": -float('inf')}
        tiers_run = False

        for tier_data in progress[model_name][prev_task].values():
            if tier_data.get("count", 0) > 0:
                tiers_run = True
                if "best_acc" in tier_data:
                    best_metrics["accuracy"] = max(
                        best_metrics["accuracy"], tier_data["best_acc"])
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

    def _check_low_data_needed(
        self, stats, progress, model, task
    ) -> Optional[ExperimentTask]:
        """
        If model performs well, test it on Low-Data regime (10%, 25%).
        """
        # Only for Vision tasks where this makes sense
        if task not in ["mnist", "cifar10", "fashion_mnist"]:
            return None

        trials = stats.get("trials", [])
        if not trials:
            return None

        # Find best trial
        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        # High bar for low-data tests
        if best_trial.accuracy < 0.90:
            return None

        # Define levels
        fractions = [0.1, 0.25]

        for frac in fractions:
            # Check if run
            study_name = f"{model}_{task}_lowdata_{frac}"

            # We can check existing trials via progress if we tracked it separately
            # But low_data usually reuses the main task name in progress dict if we aren't careful.
            # However, we should define a separate "virtual task" or just use fixed config.
            # Best check: Look for trials with data_fraction in config

            already_run = False
            for t in trials:  # These are STANDARD trials. Low data trials might be in STANDARD too?
                # Actually, we should probably run them at STANDARD tier but with data_fraction
                if t.config.get("data_fraction") == frac:
                    already_run = True
                    break

            # Also check if we created a specific study for it
            # This is harder without querying DB directly for that study.
            # But if we rely on `stats` which comes from `get_progress`, it aggregates by (model, task, tier).
            # If we run low data as same task/tier, they appear there.

            if not already_run:
                config_copy = best_trial.config.copy()
                config_copy["data_fraction"] = frac
                # Give a bit more epochs for low data? Or keep standard.
                config_copy["epochs"] = 20

                return ExperimentTask(
                    model_name=model,
                    task_name=task,
                    tier=PatientLevel.STANDARD,
                    study_name=study_name,
                    # Lower fraction = Higher priority (harder)
                    priority=85.0 - (frac * 10),
                    fixed_config=config_copy,
                    verification_of_trial_id=best_trial.trial_id
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

        # 4. EqProp Nudge Factor Ablation (Phase 5.1)
        if "eqprop" in model or "eq_prop" in model:
            current_nudge = config.get("nudge_factor", 1.0)  # Default usually 1.0
            # Test extremes
            if current_nudge != 0.1:
                ablations.append(("nudge_factor", 0.1))
            if current_nudge != 2.0:
                ablations.append(("nudge_factor", 2.0))

        # 5. Deep Hebbian Depth Ablation (Phase 5.1)
        if "hebbian" in model and "deep" in model:
            current_depth = config.get("num_layers", 100)
            if current_depth != 10:
                ablations.append(("num_layers", 10))
            if current_depth != 50:
                ablations.append(("num_layers", 50))

        # 6. Transformer Variant Ablation (Phase 5.1)
        if "transformer" in model:
            current_variant = config.get("variant", "full")
            if current_variant != "attention_only":
                ablations.append(("variant", "attention_only"))
            if current_variant != "recurrent_core":
                ablations.append(("variant", "recurrent_core"))

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
