"""
Experiment Check Functions for ScientistStrategy.

Contains helper methods that check if various experiment types are needed:
- Verification trials
- Cross-validation
- Ablation studies
- Transfer learning
- Continual learning
- Low-data regime tests
- Robustness checks
"""

import hashlib
import json
from typing import Any
from typing import Dict
from typing import Optional

from bioplausible.execution.task import ExperimentTask
from bioplausible.hyperopt import PatientLevel


def get_stats(
    progress: Dict, model: str, task: str, tier: PatientLevel
) -> Dict[str, Any]:
    """Extract stats from progress dict for a given model/task/tier."""
    try:
        return progress[model][task][tier.value]
    except KeyError:
        return {"count": 0, "best_acc": 0.0, "trials": []}


def check_verification_needed(
    stats: Dict,
    model: str,
    task: str,
    tier: PatientLevel,
    check_criterion_fn,
) -> Optional[ExperimentTask]:
    """Check if verification trials are needed for a configuration."""
    trials = stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    if not check_criterion_fn(tier, task, best_trial.accuracy):
        return None

    repeats = 0
    target_config = {
        k: v
        for k, v in best_trial.config.items()
        if k not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
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


def check_cv_needed(
    std_stats: Dict,
    progress: Dict,
    model: str,
    task: str,
) -> Optional[ExperimentTask]:
    """Check if 5-fold cross-validation is needed."""
    trials = std_stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    repeats = 0
    target_config = {
        k: v
        for k, v in best_trial.config.items()
        if k not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
    }
    target_hash = hashlib.md5(
        json.dumps(target_config, sort_keys=True).encode()
    ).hexdigest()

    for t in trials:
        t_conf = {
            k: v
            for k, v in t.config.items()
            if k
            not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
        }
        if (
            hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest()
            == target_hash
        ):
            repeats += 1

    if repeats < 3:
        return None

    cv_stats = get_stats(progress, model, task, PatientLevel.CROSS_VAL)
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


def check_continual_learning_needed(
    stats: Dict,
    progress: Dict,
    model: str,
    task: str,
) -> Optional[ExperimentTask]:
    """Schedule next steps in a Split-MNIST Continual Learning sequence."""
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
        step_stats = get_stats(progress, model, step_task, PatientLevel.STANDARD)

        if step_stats["count"] == 0:
            config_copy = {}

            if step_idx > 0:
                if previous_trial_id is None:
                    return None

                prev_task_name = steps[i - 1][0]
                prev_stats = get_stats(
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


def check_transfer_needed(
    stats: Dict,
    progress: Dict,
    model: str,
    task: str,
    curriculum,
) -> Optional[ExperimentTask]:
    """Check if transfer learning experiment should be scheduled."""
    trials = stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    if best_trial.accuracy < 0.85:
        return None

    next_task = curriculum.get_next_task(model, task, success=True)

    if not next_task or next_task == "completed_track":
        return None

    target_stats = get_stats(progress, model, next_task, PatientLevel.STANDARD)

    already_done = False
    for t in target_stats.get("trials", []):
        if t.config.get("transfer_from") == best_trial.trial_id:
            already_done = True
            break

    if not already_done:
        config_copy = best_trial.config.copy()
        config_copy["transfer_from"] = best_trial.trial_id
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


def check_low_data_needed(
    stats: Dict,
    progress: Dict,
    model: str,
    task: str,
) -> Optional[ExperimentTask]:
    """Check if low-data regime experiment should be scheduled."""
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


def check_ablation_needed(
    stats: Dict,
    progress: Dict,
    model: str,
    task: str,
    check_criterion_fn,
) -> Optional[ExperimentTask]:
    """Check if ablation study should be scheduled."""
    trials = stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    if not check_criterion_fn(PatientLevel.STANDARD, task, best_trial.accuracy):
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
            if t.config.get("is_ablation") and t.config.get("ablation_param") == param:
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


def check_robustness_needed(
    deep_stats: Dict,
    progress: Dict,
    model: str,
    task: str,
    check_criterion_fn,
) -> Optional[ExperimentTask]:
    """Check if robustness analysis should be scheduled."""
    trials = deep_stats.get("trials", [])
    if not trials:
        return None

    best_trial = max(trials, key=lambda t: t.accuracy)
    if not check_criterion_fn(PatientLevel.DEEP, task, best_trial.accuracy):
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
