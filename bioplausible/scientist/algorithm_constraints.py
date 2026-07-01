"""
Algorithm-specific hyperparameter constraints.
Prevents applying inappropriate hyperparameters to different algorithm families.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# (min, max, scale) where scale = "log" or "linear" or "int"
ALGORITHM_FAMILY_CONSTRAINTS = {
    "baseline": {  # Standard backprop
        "lr": (1e-5, 1e-2, "log"),
        "grad_clip": (0.5, 10.0, "linear"),
        "weight_decay": (0.0, 1e-2, "log"),
        "dropout": (0.0, 0.5, "linear"),
        "momentum": (0.0, 0.99, "linear"),
        "optimizer": ["sgd", "adam", "adamw", "rmsprop"],  # Categorical
        "hidden_dim": [32, 64, 128, 256, 512],
        "num_layers": (1, 4, "int"),
    },
    "eqprop": {  # Equilibrium Propagation
        "lr": (1e-6, 5e-4, "log"),  # MUCH LOWER than backprop!
        "beta": (0.01, 0.5, "linear"),
        "steps": (10, 40, "int"),
        "grad_clip": (1.0, 5.0, "linear"),  # Tighter clipping
        "nudge_type": ["output_clamping", "energy_based", "symmetric"],
        "hidden_dim": [32, 64, 128],
        "num_layers": (2, 6, "int"),
        # NO: optimizer, dropout, weight_decay, momentum (not applicable)
    },
    "hebbian": {  # Hebbian learning
        "lr": (1e-5, 1e-3, "log"),
        "contrastive_steps": (5, 30, "int"),
        "grad_clip": (1.0, 10.0, "linear"),
        "hidden_dim": [64, 128],
        "num_layers": (2, 4, "int"),
    },
    "hybrid": {  # Hybrid methods (FA, etc.)
        "lr": (1e-5, 5e-3, "log"),
        "grad_clip": (0.5, 10.0, "linear"),
        "fa_scale": (0.5, 2.0, "linear"),
        "adapt_rate": (1e-4, 1e-1, "log"),
        "hidden_dim": [64, 128, 256],
        "num_layers": (2, 5, "int"),
    },
}


def get_constrained_search_space(model_name: str) -> Dict[str, Any]:
    """
    Returns algorithm-specific hyperparameter constraints.

    Args:
        model_name: Name of the model (e.g., "Backprop Baseline", "EqProp MLP")

    Returns:
        Dictionary of hyperparameter constraints

    Example:
        >>> constraints = get_constrained_search_space("EqProp MLP")
        >>> constraints["lr"]
        (1e-6, 5e-4, "log")
        >>> "optimizer" in constraints
        False
    """
    from bioplausible.models.registry import get_model_spec

    try:
        model_spec = get_model_spec(model_name)
        family = model_spec.family.lower()
    except (KeyError, AttributeError):
        logger.warning(
            f"Could not determine family for {model_name}, using baseline constraints"
        )
        family = "baseline"

    constraints = ALGORITHM_FAMILY_CONSTRAINTS.get(
        family, ALGORITHM_FAMILY_CONSTRAINTS["baseline"]
    )

    logger.info(f"Using {family} constraints for {model_name}")
    logger.debug(f"Constraints: {list(constraints.keys())}")

    return constraints


def suggest_hyperparam(trial, param_name: str, constraint, prefix: str = ""):
    """
    Helper to suggest a hyperparameter value using Optuna trial.

    Args:
        trial: Optuna trial object
        param_name: Name of hyperparameter
        constraint: Constraint tuple (min, max, scale) or list of choices
        prefix: Optional prefix for param name in Optuna

    Returns:
        Suggested value
    """
    full_name = f"{prefix}{param_name}" if prefix else param_name

    if isinstance(constraint, list):
        # Categorical
        return trial.suggest_categorical(full_name, constraint)
    elif isinstance(constraint, tuple) and len(constraint) == 3:
        min_val, max_val, scale = constraint

        if scale == "log":
            return trial.suggest_float(full_name, min_val, max_val, log=True)
        elif scale == "int":
            return trial.suggest_int(full_name, int(min_val), int(max_val))
        else:  # linear
            return trial.suggest_float(full_name, min_val, max_val, log=False)
    else:
        raise ValueError(f"Invalid constraint format for {param_name}: {constraint}")


def create_constrained_optuna_config(
    trial,
    model_name: str,
    custom_constraints: Dict[str, Any] = None,
    task_name: str = None,
) -> Dict[str, Any]:
    """
    Create a configuration dict using algorithm-specific constraints.

    Args:
        trial: Optuna trial
        model_name: Model name
        custom_constraints: Optional dictionary of constraints
            (e.g. from failure analysis)
        task_name: Optional task name for scaling constraints

    Returns:
        Configuration dictionary
    """
    from bioplausible.hyperopt.optuna_bridge import create_optuna_space

    # Delegate to the unified Metamodel via Optuna Bridge
    # Pre-process constraints based on task difficulty
    final_constraints = custom_constraints.copy() if custom_constraints else {}

    if task_name == "cifar100":
        # Scale up model for hard task if not explicitly constrained
        if "hidden_dim" not in final_constraints:
            final_constraints["hidden_dim"] = [128, 256, 512, 1024]
        if "num_layers" not in final_constraints:
            final_constraints["min_num_layers"] = 3
            final_constraints["max_num_layers"] = 8

    # Apply safety caps (e.g. from OOM analysis)
    if "max_hidden_dim" in final_constraints:
        max_dim = final_constraints["max_hidden_dim"]
        if "hidden_dim" in final_constraints and isinstance(
            final_constraints["hidden_dim"], list
        ):
            # Filter choices
            filtered = [d for d in final_constraints["hidden_dim"] if d <= max_dim]
            if not filtered:
                filtered = [max_dim]  # Fallback to max allowed
            final_constraints["hidden_dim"] = filtered

    return create_optuna_space(
        trial=trial,
        model_name=model_name,
        constraints=final_constraints,
        task_name=task_name,
    )
