import hashlib
import json
import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from bioplausible.hyperopt import PatientLevel
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible.scientist.curriculum import CurriculumManager
from bioplausible.scientist.dashboard import DASHBOARD
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
        PatientLevel.SMOKE: lambda acc: acc > 0.12,  # Beat random (0.10) slightly
        PatientLevel.SHALLOW: lambda acc: acc > 0.30,  # Relaxed for early feedback
        PatientLevel.STANDARD: lambda acc: acc > 0.60,
        PatientLevel.CROSS_VAL: lambda acc: True,  # CV just needs to run 5 times
        PatientLevel.DEEP: lambda acc: acc > 0.80,  # Deep bar
    }

    TASK_WEIGHTS = {
        "digits": 0.50,  # Fastest proxy (Tiny) - Boosted for early filtering
        "usps": 0.45,  # Fast proxy (Small) - Boosted
        "kmnist": 0.35,  # Boosted
        "mnist": 0.30,
        "cartpole": 0.40,  # RL Smoke test
        "pendulum": 0.35,  # RL Intermediate
        "acrobot": 0.30,  # RL Hard
        "fashion_mnist": 0.25,
        "svhn": 0.20,
        "char_ngram": 0.10,
        "tiny_shakespeare": 0.05,
        "cifar10": 0.15,
        "cifar100": 0.10,
    }
    TASK_GROUPS = {
        "vision": [
            "digits",
            "usps",
            "kmnist",
            "mnist",
            "fashion_mnist",
            "svhn",
            "cifar10",
            "cifar100",
        ],
        "lm": ["char_ngram", "tiny_shakespeare"],
        "rl": ["cartpole", "pendulum", "acrobot"],
    }

    def __init__(
        self,
        state: ExperimentState,
        decision_logger: Optional[DecisionLogger] = None,
        task_filter: Optional[str] = None,
        tier_limit: Optional[str] = None,
    ):
        self.state = state
        self.decision_logger = decision_logger
        self.task_filter = task_filter
        self.tier_limit = tier_limit.lower() if tier_limit else None
        self._logged_events: Set[str] = set()
        self.curriculum = CurriculumManager()

        self.TIER_ORDER = {
            PatientLevel.SMOKE: 0,
            PatientLevel.SHALLOW: 1,
            PatientLevel.STANDARD: 2,
            PatientLevel.DEEP: 3,
            PatientLevel.CROSS_VAL: 4,
        }

    def _log(
        self,
        key: str,
        event_type: str,
        desc: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if key not in self._logged_events:
            if self.decision_logger:
                self.decision_logger.log_decision(event_type, desc, meta)
            self._logged_events.add(key)

        # Update Dashboard Insight
        DASHBOARD.set_insight(desc)

    def _check_criterion(self, tier: PatientLevel, task: str, acc: float) -> bool:
        """
        Check if accuracy meets the success criterion for a given tier and task.
        Allows task-specific overrides (e.g., lower threshold for CIFAR-100).
        """
        # Task-specific overrides
        if task == "cifar100":
            if tier == PatientLevel.SMOKE:
                return acc > 0.05  # 5x random chance
            elif tier == PatientLevel.SHALLOW:
                return acc > 0.15
            elif tier == PatientLevel.STANDARD:
                return acc > 0.30
            elif tier == PatientLevel.DEEP:
                return acc > 0.50

        # Fast Fail for Easy Tasks
        if task in ["digits", "usps"]:
            if tier == PatientLevel.SMOKE:
                return acc > 0.50  # Must be much better than random
            elif tier == PatientLevel.SHALLOW:
                return acc > 0.80

        if task == "tiny_shakespeare":
            # LM uses perplexity mostly but acc is tracked too.
            # Character-level LM accuracy is usually lower.
            if tier == PatientLevel.SMOKE:
                return acc > 0.30
            elif tier == PatientLevel.STANDARD:
                return acc > 0.45

        return self.CRITERIA[tier](acc)

    def _matches_filter(self, task: str) -> bool:
        if not self.task_filter or self.task_filter == "all":
            return True
        if self.task_filter == task:
            return True
        if self.task_filter in self.TASK_GROUPS:
            return task in self.TASK_GROUPS[self.task_filter]
        return False

    def generate_candidates(self) -> List[ExperimentTask]:
        """
        Generates a list of all possible valid experiments based on current state.
        """
        progress = self.state.get_progress()
        candidates: List[ExperimentTask] = []

        # Analyze failures to generate constraints
        failure_constraints = self._analyze_failures(progress)
        self._apply_failure_logging(failure_constraints)

        # Analyze fragility (High Accuracy but Low Robustness)
        fragility_constraints = self._analyze_fragility()
        if fragility_constraints:
            # Merge constraints
            for m, c in fragility_constraints.items():
                if m not in failure_constraints:
                    failure_constraints[m] = {}
                failure_constraints[m].update(c)
                self._log(
                    f"fragile_constraint_{m}",
                    "ROBUSTNESS_ENFORCED",
                    f"Model {m} is fragile. Enforcing regularization.",
                    c,
                )

        # Analyze saturation (Tasks that are "solved")
        saturated_tasks = self._analyze_saturation(progress)
        self._apply_saturation_logging(saturated_tasks)

        for spec in MODEL_REGISTRY:
            tasks = self._resolve_tasks(spec.task_compat, spec.name)

            for task in tasks:
                if not self._should_consider_task(
                    spec.name, task, progress, saturated_tasks
                ):
                    continue

                candidates.extend(
                    self._generate_candidates_for_task(
                        spec.name, task, progress, failure_constraints
                    )
                )

        self._filter_by_tier_limit(candidates)
        self._apply_prioritization(candidates)

        return candidates

    def _should_consider_task(
        self, model_name: str, task: str, progress: Dict, saturated_tasks: Dict
    ) -> bool:
        """Check if a task should be considered for candidate generation."""
        if not self._matches_filter(task):
            return False

        if model_name in saturated_tasks and task in saturated_tasks[model_name]:
            return False

        if not self._check_curriculum(progress, model_name, task):
            return False

        return True

    def _generate_candidates_for_task(
        self, model: str, task: str, progress: Dict, failure_constraints: Dict
    ) -> List[ExperimentTask]:
        """Generate candidates for a specific model/task pair across tiers."""
        candidates = []

        # 1. SMOKE
        smoke_task = self._check_smoke_tier(model, task, progress)
        if smoke_task:
            candidates.append(smoke_task)
            # If we are scheduling smoke, we generally don't schedule higher tiers yet
            # unless smoke is passed.
            # The logic below checks if smoke is PASSED before generating higher.
            # But if smoke_task is generated, it means we need to run it.
            # However, existing logic allowed fall-through if smoke stats existed but failed criterion?
            # No, if smoke_stats < 3, we generate smoke task and CONTINUE loop in original code.
            return candidates

        smoke_stats = self._get_stats(progress, model, task, PatientLevel.SMOKE)
        if not self._check_criterion(PatientLevel.SMOKE, task, smoke_stats["best_acc"]):
            # Retry chance for failed smoke
            if random.random() < 0.01:
                candidates.append(
                    self._make_task(model, task, PatientLevel.SMOKE, 10.0)
                )
            return candidates

        # 2. SHALLOW
        shallow_task = self._check_shallow_tier(
            model, task, progress, smoke_stats, failure_constraints
        )
        if shallow_task:
            candidates.append(shallow_task)
            return candidates

        shallow_stats = self._get_stats(progress, model, task, PatientLevel.SHALLOW)
        if not self._check_criterion(
            PatientLevel.SHALLOW, task, shallow_stats["best_acc"]
        ):
            self._log(
                f"stagnated_shallow_{model}_{task}",
                "STAGNATION",
                f"Model {model} failed Shallow Tier on {task} (Acc: {shallow_stats['best_acc']:.2%}). Stopping.",
                {"best_acc": shallow_stats["best_acc"]},
            )
            return candidates

        # 3. STANDARD
        candidates.extend(
            self._generate_standard_candidates(
                model, task, progress, shallow_stats, failure_constraints
            )
        )

        # If we just generated a standard exploration task (not verification/transfer/etc),
        # we might stop here? Original code had `continue` if std_stats["count"] < 20.
        # Let's check if we generated a main standard task.
        std_stats = self._get_stats(progress, model, task, PatientLevel.STANDARD)
        if std_stats["count"] < 20:
            return candidates

        if not self._check_criterion(
            PatientLevel.STANDARD, task, std_stats["best_acc"]
        ):
            return candidates

        # 4. DEEP
        candidates.extend(
            self._generate_deep_candidates(
                model, task, progress, std_stats, failure_constraints
            )
        )

        return candidates

    def _check_smoke_tier(
        self, model: str, task: str, progress: Dict
    ) -> Optional[ExperimentTask]:
        smoke_stats = self._get_stats(progress, model, task, PatientLevel.SMOKE)
        if smoke_stats["count"] < 3:
            if smoke_stats["count"] == 0:
                self._log(
                    f"smoke_{model}_{task}",
                    "NEW_HYPOTHESIS",
                    f"Starting initial investigation (Smoke Test) for {model} on {task}.",
                )

            p = 100.0 if smoke_stats["count"] == 0 else 80.0
            return self._make_task(model, task, PatientLevel.SMOKE, p)
        return None

    def _check_shallow_tier(
        self,
        model: str,
        task: str,
        progress: Dict,
        smoke_stats: Dict,
        failure_constraints: Dict,
    ) -> Optional[ExperimentTask]:
        shallow_stats = self._get_stats(progress, model, task, PatientLevel.SHALLOW)
        if shallow_stats["count"] < 10:
            # Tuned Priority
            base_p = 50.0 + (smoke_stats["best_acc"] * 30.0)
            if shallow_stats["count"] == 0:
                self._log(
                    f"shallow_{model}_{task}",
                    "PROMOTION",
                    f"Promoting {model} to Shallow Tier (Passed Smoke Test with {smoke_stats['best_acc']:.2%}).",
                )
                base_p += 10.0

            model_constraints = failure_constraints.get(model, {})
            task_obj = self._make_task(model, task, PatientLevel.SHALLOW, base_p)
            if model_constraints:
                task_obj.constraints = model_constraints
            return task_obj
        return None

    def _generate_standard_candidates(
        self,
        model: str,
        task: str,
        progress: Dict,
        shallow_stats: Dict,
        failure_constraints: Dict,
    ) -> List[ExperimentTask]:
        candidates = []
        std_stats = self._get_stats(progress, model, task, PatientLevel.STANDARD)

        # Verification
        v_task = self._check_verification_needed(
            std_stats, model, task, PatientLevel.STANDARD
        )
        if v_task:
            self._log(
                f"verify_std_{model}_{task}",
                "VERIFICATION",
                f"Verifying best result for {model} (Standard Tier).",
            )
            candidates.append(v_task)

        # Low Data
        ld_task = self._check_low_data_needed(std_stats, progress, model, task)
        if ld_task:
            self._log(
                f"low_data_{model}_{task}",
                "LOW_DATA_REGIME",
                f"Scheduling Low-Data experiment ({ld_task.fixed_config['data_fraction']:.0%}) for {model}.",  # type: ignore
            )
            candidates.append(ld_task)

        # Ablation
        ab_task = self._check_ablation_needed(std_stats, progress, model, task)
        if ab_task:
            self._log(
                f"ablation_{model}_{task}_{ab_task.ablation_param}",
                "ABLATION_STUDY",
                f"Scheduling ablation study for {model} to verify components.",
                {"param": ab_task.ablation_param},
            )
            candidates.append(ab_task)

        # Continual Learning
        cl_task = self._check_continual_learning_needed(
            std_stats, progress, model, task
        )
        if cl_task:
            self._log(
                f"cl_{model}_{task}_{cl_task.continual_step}",
                "CONTINUAL_LEARNING",
                f"Attempting Continual Learning Step {cl_task.continual_step} for {model}.",
            )
            candidates.append(cl_task)

        # Transfer Learning
        tf_task = self._check_transfer_needed(std_stats, progress, model, task)
        if tf_task:
            self._log(
                f"transfer_{model}_{task}",
                "TRANSFER_LEARNING",
                f"Attempting Transfer Learning from {task} for {model}.",
            )
            candidates.append(tf_task)

        # Cross Validation
        cv_task = self._check_cv_needed(std_stats, progress, model, task)
        if cv_task:
            self._log(
                f"cv_{model}_{task}",
                "CROSS_VALIDATION",
                f"Running 5-Fold Cross-Validation for {model} to confirm stability.",
            )
            candidates.append(cv_task)

        # Main Standard Exploration
        if std_stats["count"] < 20:
            base_p = 60.0 + (shallow_stats["best_acc"] * 40.0)

            if std_stats["count"] == 0:
                self._log(
                    f"standard_{model}_{task}",
                    "PROMOTION",
                    f"Promoting {model} to Standard Tier (Passed Shallow with {shallow_stats['best_acc']:.2%}).",
                )

            if std_stats["count"] > 15:
                base_p -= 10.0

            refine_constraints = self._refine_search_space(
                progress, model, task, PatientLevel.SHALLOW
            )
            fail_constraints = failure_constraints.get(model, {})

            final_constraints = {}
            if refine_constraints:
                self._log(
                    f"refine_std_{model}_{task}",
                    "REFINEMENT",
                    f"Refining search space for Standard Tier based on Shallow results.",
                    refine_constraints,
                )
                final_constraints.update(refine_constraints)
            if fail_constraints:
                final_constraints.update(fail_constraints)

            task_obj = self._make_task(model, task, PatientLevel.STANDARD, base_p)
            if final_constraints:
                task_obj.constraints = final_constraints

            candidates.append(task_obj)

        return candidates

    def _generate_deep_candidates(
        self,
        model: str,
        task: str,
        progress: Dict,
        std_stats: Dict,
        failure_constraints: Dict,
    ) -> List[ExperimentTask]:
        candidates = []
        deep_stats = self._get_stats(progress, model, task, PatientLevel.DEEP)

        # Robustness
        r_task = self._check_robustness_needed(deep_stats, progress, model, task)
        if r_task:
            self._log(
                f"robust_{model}_{task}",
                "ROBUSTNESS_CHECK",
                f"Triggering Robustness Analysis for {model} due to high Deep Tier performance.",
            )
            candidates.append(r_task)

        # Verification
        v_task = self._check_verification_needed(
            deep_stats, model, task, PatientLevel.DEEP
        )
        if v_task:
            candidates.append(v_task)

        # Main Deep Exploration
        if deep_stats["count"] < 5:
            if deep_stats["count"] == 0:
                self._log(
                    f"deep_{model}_{task}",
                    "PROMOTION",
                    f"Promoting {model} to Deep Tier (Passed Standard with {std_stats['best_acc']:.2%}).",
                )

            p = 20.0 + (std_stats["best_acc"] * 50.0)

            refine_constraints = self._refine_search_space(
                progress, model, task, PatientLevel.STANDARD
            )
            fail_constraints = failure_constraints.get(model, {})

            final_constraints = {}
            if refine_constraints:
                self._log(
                    f"refine_deep_{model}_{task}",
                    "REFINEMENT",
                    f"Refining search space for Deep Tier based on Standard results.",
                    refine_constraints,
                )
                final_constraints.update(refine_constraints)
            if fail_constraints:
                final_constraints.update(fail_constraints)

            task_obj = self._make_task(model, task, PatientLevel.DEEP, p)
            if final_constraints:
                task_obj.constraints = final_constraints

            candidates.append(task_obj)

        return candidates

    def _apply_failure_logging(self, failure_constraints: Dict) -> None:
        if failure_constraints:
            for model, constraints in failure_constraints.items():
                self._log(
                    f"fail_constraint_{model}",
                    "CONSTRAINT_APPLIED",
                    f"High failure rate detected for {model}. Restricting search space.",
                    constraints,
                )

    def _apply_saturation_logging(self, saturated_tasks: Dict) -> None:
        if saturated_tasks:
            for model, tasks in saturated_tasks.items():
                for t in tasks:
                    self._log(
                        f"saturation_{model}_{t}",
                        "SATURATION",
                        f"Task {t} is saturated (solved) for {model}. Skipping.",
                    )

    def _filter_by_tier_limit(self, candidates: List[ExperimentTask]) -> None:
        if self.tier_limit:
            limit_level = -1
            for tier, level in self.TIER_ORDER.items():
                if tier.value == self.tier_limit:
                    limit_level = level
                    break

            if limit_level != -1:
                # We can't modify list in place while iterating easily, so create new list
                # Actually modifying the list passed by reference
                # candidates[:] = [c for c in candidates if ...]
                candidates[:] = [
                    c
                    for c in candidates
                    if self.TIER_ORDER.get(c.tier, 999) <= limit_level
                ]

    def _apply_prioritization(self, candidates: List[ExperimentTask]) -> None:
        # Task Rebalancing & Future Value
        for c in candidates:
            weight = self.TASK_WEIGHTS.get(c.task_name, 0.10)
            future_boost = self._calculate_future_boost(c.task_name, weight)
            effective_weight = weight + future_boost
            c.priority *= effective_weight * 5.0

        # Diversity Penalty (Task)
        recent_tasks = self.state.get_recent_tasks(limit=10)
        task_counts: Dict[str, int] = {}
        for t in recent_tasks:
            task_counts[t] = task_counts.get(t, 0) + 1

        # Diversity Penalty (Model)
        recent_models = self.state.get_recent_models(limit=10)
        model_counts: Dict[str, int] = {}
        for m in recent_models:
            model_counts[m] = model_counts.get(m, 0) + 1

        for c in candidates:
            # Task penalty
            t_count = task_counts.get(c.task_name, 0)
            if t_count > 0:
                penalty = 0.9**t_count
                c.priority *= penalty

            # Model penalty
            m_count = model_counts.get(c.model_name, 0)
            if m_count > 0:
                penalty = 0.8**m_count
                c.priority *= penalty

    def _calculate_future_boost(self, task_name: str, current_weight: float) -> float:
        future_boost = 0.0
        for track_name, track_tasks in self.curriculum.TRACKS.items():
            if task_name in track_tasks:
                idx = track_tasks.index(task_name)
                for forward_idx in range(idx + 1, len(track_tasks)):
                    future_task = track_tasks[forward_idx]
                    future_weight = self.TASK_WEIGHTS.get(future_task, 0.10)

                    if future_weight > current_weight:
                        distance = forward_idx - idx
                        boost = (future_weight - current_weight) * (0.9**distance)
                        if boost > future_boost:
                            future_boost = boost
        return future_boost

    def _refine_search_space(
        self, progress, model, task, source_tier
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze successful trials from source_tier to refine search space for next tier.
        """
        stats = self._get_stats(progress, model, task, source_tier)
        trials = stats.get("trials", [])

        if len(trials) < 3:
            return None

        trials.sort(key=lambda x: x.accuracy, reverse=True)
        top_n = max(3, len(trials) // 2)
        top_trials = trials[:top_n]

        if top_trials[0].accuracy < 0.2:
            return None

        constraints = {}

        lrs = [t.config["lr"] for t in top_trials if "lr" in t.config]
        if lrs:
            min_lr = min(lrs)
            max_lr = max(lrs)
            constraints["min_lr"] = min_lr * 0.5
            constraints["max_lr"] = max_lr * 2.0

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

    def _analyze_fragility(self) -> Dict[str, Dict[str, Any]]:
        """
        Identify models that perform well but break easily, and suggest constraints.
        """
        constraints = {}
        if hasattr(self.state, "get_fragile_models"):
            fragile_models = self.state.get_fragile_models()
            for model, score in fragile_models.items():
                constraints[model] = {
                    "min_weight_decay": 1e-4,
                    "min_dropout": 0.2,
                    "use_spectral_norm": True,
                }
        return constraints

    def _analyze_failures(self, progress) -> Dict[str, Dict[str, Any]]:
        """
        Analyze failure rates to suggest constraints.
        Returns: Dict[model_name, constraint_dict]
        """
        constraints = {}

        # 1. Query FailureTracker via State for Hard Failures
        if hasattr(self.state, "get_failure_analysis"):
            try:
                analysis = self.state.get_failure_analysis()
                recommendations = analysis.get("recommendations", [])

                for rec in recommendations:
                    if rec.get("issue") == "High NaN failure rate":
                        affected = rec.get("affected_models", [])
                        for model in affected:
                            if model not in constraints:
                                constraints[model] = {}
                            # Aggressive restriction
                            constraints[model]["max_lr"] = 0.001
                            constraints[model]["max_beta"] = 0.1

                    elif rec.get("issue") == "Out of memory errors":
                        # Constrain models to prevent OOM loop
                        # Ideally check affected models, but OOM often crashes system so we might blame last run
                        # If affected_models is empty, apply to all active models?
                        # Let's trust the FailureTracker to have identified context if possible.
                        # If not, apply to all models in progress.
                        affected = rec.get("affected_models", [])
                        if not affected:
                            # Fallback: Apply to everything if systemic OOM
                            affected = list(progress.keys())

                        for model in affected:
                            if model not in constraints:
                                constraints[model] = {}
                            constraints[model]["max_batch_size"] = 32
                            constraints[model][
                                "max_hidden_dim"
                            ] = 256  # Prevent aggressive scaling

                    elif rec.get("issue") == "Early Training Instability":
                        # If we knew which models, we'd constrain them.
                        pass

            except Exception as e:
                logger.warning(f"Failed to query failure analysis: {e}")

        # 2. Analyze Progress for Soft Failures (Divergence/No Learning)
        for model, task_data in progress.items():
            total = 0
            failures = 0
            for task, tier_data in task_data.items():
                for tier, stats in tier_data.items():
                    trials = stats.get("trials", [])
                    for t in trials:
                        total += 1
                        if (
                            t.final_loss > 100 or t.accuracy < 0.11
                        ):  # Divergence or random chance
                            failures += 1

            if total > 5 and (failures / total) > 0.3:
                # If not already constrained more strictly
                if model not in constraints:
                    constraints[model] = {}
                    constraints[model]["max_lr"] = 0.005
                    constraints[model]["max_beta"] = 0.5

        return constraints

    def _analyze_saturation(self, progress) -> Dict[str, List[str]]:
        """
        Identify tasks that are effectively "solved" (saturated) for a given model.
        Returns: Dict[model, List[task_name]]
        """
        saturation = {}

        for model, task_data in progress.items():
            solved_tasks = []

            # Check for direct saturation
            for task, tiers in task_data.items():
                best_acc = 0.0
                for tier_stats in tiers.values():
                    best_acc = max(best_acc, tier_stats.get("best_acc", 0.0))

                # Dynamic saturation thresholds
                threshold = 0.99
                if task == "digits":
                    threshold = 0.98
                elif task == "mnist":
                    threshold = 0.99
                elif task == "fashion_mnist":
                    threshold = 0.94

                if best_acc > threshold:
                    solved_tasks.append(task)

            # Implicit Saturation: If a harder task is solved, easier ones are "solved"
            if "mnist" in solved_tasks:
                if "digits" not in solved_tasks:
                    solved_tasks.append("digits")
                if "usps" not in solved_tasks:
                    solved_tasks.append("usps")

            if "fashion_mnist" in solved_tasks:
                if "mnist" not in solved_tasks:
                    solved_tasks.append("mnist")
                if "kmnist" not in solved_tasks:
                    solved_tasks.append("kmnist")

            if solved_tasks:
                saturation[model] = solved_tasks

        return saturation

    def plan_next(self) -> Optional[ExperimentTask]:
        """
        Scans all possibilities and returns the highest priority experiment.
        """
        candidates = self.generate_candidates()

        if not candidates:
            return None

        # Standard Tier Calibration
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
                    f"Calibration Mode Active: Boosted Standard Tier candidates (Count: {total_standard_trials}/50)"
                )

        candidates.sort(key=lambda x: x.priority + random.uniform(0, 5), reverse=True)
        return candidates[0]

    def plan_batch(self, batch_size: int) -> List[ExperimentTask]:
        """
        Generate a batch of unique, high-priority experiments.
        """
        candidates = self.generate_candidates()
        if not candidates:
            return []

        # Add noise to priority for diversity
        for c in candidates:
            c.priority += random.uniform(0, 5)

        candidates.sort(key=lambda x: x.priority, reverse=True)

        batch = []
        seen = set()
        for c in candidates:
            key = f"{c.model_name}_{c.task_name}_{c.tier.value}"
            if key not in seen:
                batch.append(c)
                seen.add(key)
            if len(batch) >= batch_size:
                break

        return batch

    def _resolve_tasks(self, task_compat: List[str], model_name: str = "") -> List[str]:
        """
        Convert compatibility list to specific runnable tasks.
        Uses CurriculumManager to refine choices.
        """
        if not task_compat:
            initial = self.curriculum.get_initial_task(model_name)
            return [initial] if initial else ["mnist"]

        resolved = []
        for t in task_compat:
            if t in self.curriculum.TRACKS:
                resolved.extend(self.curriculum.TRACKS[t])
            else:
                resolved.append(t)
        return list(set(resolved))

    def _check_curriculum(self, progress: Dict, model_name: str, task: str) -> bool:
        """
        Check if we are allowed to run this task based on curriculum.
        """
        track = None
        for t_list in self.curriculum.TRACKS.values():
            if task in t_list:
                track = t_list
                break

        if not track:
            return True

        try:
            curr_idx = track.index(task)
        except ValueError:
            return True

        if curr_idx == 0:
            return True

        prev_task = track[curr_idx - 1]

        if model_name not in progress or prev_task not in progress[model_name]:
            return False

        best_metrics = {"accuracy": 0.0, "reward": -float("inf")}
        tiers_run = False

        for tier_data in progress[model_name][prev_task].values():
            if tier_data.get("count", 0) > 0:
                tiers_run = True
                if "best_acc" in tier_data:
                    best_metrics["accuracy"] = max(
                        best_metrics["accuracy"], tier_data["best_acc"]
                    )

        if not tiers_run:
            return False

        if PromotionGate.check_promotion(prev_task, best_metrics):
            return True
        else:
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
        """
        if task != "mnist":
            return None

        if stats["count"] == 0 or stats["best_acc"] < 0.95:
            return None

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
                config_copy = {}

                if step_idx > 0:
                    if previous_trial_id is None:
                        return None

                    prev_task_name = steps[i - 1][0]
                    prev_stats = self._get_stats(
                        progress, model, prev_task_name, PatientLevel.STANDARD
                    )
                    best_prev = max(prev_stats["trials"], key=lambda t: t.accuracy)
                    config_copy = best_prev.config.copy()
                    config_copy["transfer_from"] = best_prev.trial_id
                    config_copy["freeze_layers"] = False
                else:
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

            best_step_trial = max(step_stats["trials"], key=lambda t: t.accuracy)
            if best_step_trial.accuracy < 0.80:
                return None

            previous_trial_id = best_step_trial.trial_id

        return None

    def _check_transfer_needed(
        self, stats, progress, model, task
    ) -> Optional[ExperimentTask]:
        """
        If a model masters a base task, try transferring to the next task in the curriculum.
        """
        # 1. Check if current task is mastered
        trials = stats.get("trials", [])
        if not trials:
            return None

        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        # Mastery Threshold (could be task-specific, using simple heuristic for now)
        if best_trial.accuracy < 0.85:
            return None

        # 2. Identify Next Task in Curriculum
        # We pass success=True because we only transfer if successful
        next_task = self.curriculum.get_next_task(model, task, success=True)

        if not next_task or next_task == "completed_track":
            return None

        # 3. Check if transfer already attempted
        target_stats = self._get_stats(
            progress, model, next_task, PatientLevel.STANDARD
        )

        already_done = False
        for t in target_stats.get("trials", []):
            if t.config.get("transfer_from") == best_trial.trial_id:
                already_done = True
                break

        if not already_done:
            config_copy = best_trial.config.copy()
            config_copy["transfer_from"] = best_trial.trial_id
            # Default to freezing for transfer, though fine-tuning is also valid.
            config_copy["freeze_layers"] = True

            return ExperimentTask(
                model_name=model,
                task_name=next_task,
                tier=PatientLevel.STANDARD,
                study_name=f"{model}_{next_task}_transfer",
                priority=92.0,
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
        if task not in ["mnist", "cifar10", "fashion_mnist"]:
            return None

        trials = stats.get("trials", [])
        if not trials:
            return None

        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        if best_trial.accuracy < 0.90:
            return None

        fractions = [0.1, 0.25]

        for frac in fractions:
            study_name = f"{model}_{task}_lowdata_{frac}"

            already_run = False
            for t in trials:
                if t.config.get("data_fraction") == frac:
                    already_run = True
                    break

            if not already_run:
                config_copy = best_trial.config.copy()
                config_copy["data_fraction"] = frac
                config_copy["epochs"] = 20

                return ExperimentTask(
                    model_name=model,
                    task_name=task,
                    tier=PatientLevel.STANDARD,
                    study_name=study_name,
                    priority=85.0 - (frac * 10),
                    fixed_config=config_copy,
                    verification_of_trial_id=best_trial.trial_id,
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

        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        if not self._check_criterion(PatientLevel.STANDARD, task, best_trial.accuracy):
            return None

        ablations = []
        config = best_trial.config

        if "symmetric_weights" in config:
            ablations.append(("symmetric_weights", not config["symmetric_weights"]))

        if config.get("beta", 0.0) > 0.0:
            ablations.append(("beta", 0.0))

        if config.get("use_top_down", False):
            ablations.append(("use_top_down", False))

        if "eqprop" in model or "eq_prop" in model:
            current_nudge = config.get("nudge_factor", 1.0)
            if current_nudge != 0.1:
                ablations.append(("nudge_factor", 0.1))
            if current_nudge != 2.0:
                ablations.append(("nudge_factor", 2.0))

        if "hebbian" in model and "deep" in model:
            current_depth = config.get("num_layers", 100)
            if current_depth != 10:
                ablations.append(("num_layers", 10))
            if current_depth != 50:
                ablations.append(("num_layers", 50))

        if "transformer" in model:
            current_variant = config.get("variant", "full")
            if current_variant != "attention_only":
                ablations.append(("variant", "attention_only"))
            if current_variant != "recurrent_core":
                ablations.append(("variant", "recurrent_core"))

        for param, val in ablations:
            already_run = False
            for t in trials:
                if (
                    t.config.get("is_ablation")
                    and t.config.get("ablation_param") == param
                ):
                    already_run = True
                    break

            if not already_run:
                config_copy = config.copy()
                config_copy[param] = val
                config_copy["is_ablation"] = True
                config_copy["ablation_param"] = param

                priority = 80.0

                return ExperimentTask(
                    model_name=model,
                    task_name=task,
                    tier=PatientLevel.STANDARD,
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

        best_trial = max(trials, key=lambda t: t.accuracy)
        if not self._check_criterion(PatientLevel.DEEP, task, best_trial.accuracy):
            return None

        for t in trials:
            if t.config.get("is_robustness_check"):
                return None

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

        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

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
            return None

        cv_stats = self._get_stats(progress, model, task, PatientLevel.CROSS_VAL)
        cv_trials = cv_stats.get("trials", [])

        completed_folds = set()
        for t in cv_trials:
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

        for fold in range(5):
            if fold not in completed_folds:
                config_copy = best_trial.config.copy()
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

        trials.sort(key=lambda x: x.accuracy, reverse=True)
        best_trial = trials[0]

        if not self._check_criterion(tier, task, best_trial.accuracy):
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
