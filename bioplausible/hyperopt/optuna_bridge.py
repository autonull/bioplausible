"""
Optuna Bridge for Bioplausible

Maps ModelSpec and SearchSpace definitions to Optuna suggest_* calls.
Replaces custom evolution code with Optuna's proven algorithms.
"""

from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import NSGAIISampler, TPESampler

from bioplausible.models.registry import ModelSpec, get_model_spec


def scalarize_objectives(
    accuracy: float, param_count: float, iteration_time: float
) -> float:
    """
    Scalarize multi-objectives into single score with priorities:
    #1 Maximize accuracy (weight: 1.0)
    #2 Minimize param count (weight: 0.01)
    #3 Minimize iteration time (weight: 0.001)

    Args:
        accuracy: Test accuracy (0-1)
        param_count: Model parameters in millions
        iteration_time: Time per iteration in seconds

    Returns:
        Scalar score (higher is better)
    """
    score = (
        accuracy * 1.0  # Primary: maximize accuracy
        - param_count * 0.01  # Secondary: minimize params
        - iteration_time * 0.001  # Tertiary: minimize time
    )
    return score


def create_optuna_space(
    trial: optuna.Trial,
    model_name: str,
    constraints: Optional[Dict[str, Any]] = None,
    evaluation_config: Optional[Any] = None,  # EvaluationConfig
) -> Dict[str, Any]:
    """
    Create Optuna hyperparameter space from SearchSpace definition.

    Args:
        trial: Optuna trial object
        model_name: Name of model from ModelRegistry
        constraints: Optional constraints (max_layers, max_hidden, etc.)
        evaluation_config: Optional EvaluationConfig for patience-based constraints

    Returns:
        Config dictionary with sampled hyperparameters
    """
    # Get search space - this is the single source of truth
    from .search_space import get_search_space

    space = get_search_space(model_name)
    config = {}

    # Merge constraints from evaluation_config if provided
    if evaluation_config:
        if constraints is None:
            constraints = {}
        if hasattr(evaluation_config, "max_hidden_dim"):
            constraints["max_hidden"] = evaluation_config.max_hidden_dim
        if hasattr(evaluation_config, "max_layers"):
            constraints["max_layers"] = evaluation_config.max_layers
        if hasattr(evaluation_config, "epochs"):
            config["epochs"] = evaluation_config.epochs

    # Iterate through search space parameters
    for param_name, param_spec in space.params.items():
        # Skip if already set (e.g., epochs from evaluation_config)
        if param_name in config:
            continue

        if isinstance(param_spec, tuple):
            # Continuous or integer range: (min, max, scale)
            min_val, max_val, scale = param_spec

            # Apply constraints
            if (
                param_name == "hidden_dim"
                and constraints
                and "max_hidden" in constraints
            ):
                max_val = min(max_val, constraints["max_hidden"])
            elif (
                param_name == "num_layers"
                and constraints
                and "max_layers" in constraints
            ):
                max_val = min(max_val, constraints["max_layers"])
            elif param_name == "steps" and constraints and "max_steps" in constraints:
                max_val = min(max_val, constraints["max_steps"])

            if scale == "log":
                config[param_name] = trial.suggest_float(
                    param_name, min_val, max_val, log=True
                )
            elif scale == "int":
                config[param_name] = trial.suggest_int(
                    param_name, int(min_val), int(max_val)
                )
            else:  # linear
                config[param_name] = trial.suggest_float(param_name, min_val, max_val)

        elif isinstance(param_spec, list):
            # Categorical choice
            config[param_name] = trial.suggest_categorical(param_name, param_spec)
        else:
            # Fixed value
            config[param_name] = param_spec

    return config


def create_study(
    model_names: List[str],
    n_objectives: int = 2,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    use_pruning: bool = True,
    sampler_name: str = "tpe",
    evaluation_config: Optional[Any] = None,  # EvaluationConfig from eval_tiers
    mode: str = "pareto",  # "pareto" or "scalarized"
) -> optuna.Study:
    """
    Create an Optuna study for hyperparameter optimization.

    Args:
        model_names: List of model names to optimize
        n_objectives: Number of objectives (1=single, 2=multi like accuracy+loss)
        storage: Storage URL (e.g., "sqlite:///optuna.db"). None for in-memory.
        study_name: Name for the study
        use_pruning: Whether to use automatic pruning
        sampler_name: "tpe", "nsga2", or "random"
        evaluation_config: Optional EvaluationConfig for patience-based settings
        mode: "pareto" for multi-objective Pareto frontier, "scalarized" for weighted single objective

    Returns:
        Optuna study object
    """
    # Override pruning from evaluation_config if provided
    if evaluation_config and hasattr(evaluation_config, "use_pruning"):
        use_pruning = evaluation_config.use_pruning

    # Direction: maximize accuracy, minimize loss/params/time
    # For scalarized mode, force n_objectives=1
    if mode == "scalarized":
        directions = ["maximize"]  # Maximize scalarized score
        n_objectives = 1
    elif n_objectives == 1:
        directions = ["maximize"]
    elif n_objectives == 2:
        directions = ["maximize", "minimize"]  # accuracy, loss
    elif n_objectives == 3:
        directions = ["maximize", "minimize", "minimize"]  # accuracy, params, time
    else:
        directions = ["maximize"] + ["minimize"] * (n_objectives - 1)

    # Sampler selection with config
    n_startup = 10  # default
    if evaluation_config and hasattr(evaluation_config, "n_startup_trials"):
        n_startup = evaluation_config.n_startup_trials

    if sampler_name == "nsga2":
        sampler = NSGAIISampler()
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler()
    else:  # TPE
        sampler = TPESampler(multivariate=True, n_startup_trials=n_startup)

    # Pruner selection
    pruner = HyperbandPruner() if use_pruning else MedianPruner()

    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    # Store mode metadata
    study.set_user_attr("mode", mode)

    return study


def get_pareto_trials(study: optuna.Study) -> List[optuna.trial.FrozenTrial]:
    """
    Get Pareto frontier trials from a multi-objective study.

    Args:
        study: Optuna study

    Returns:
        List of trials on the Pareto frontier
    """
    if len(study.directions) == 1:
        # Single objective - just return best trial
        return [study.best_trial]

    # Multi-objective - get Pareto front
    return study.best_trials


def trial_to_metrics(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    """
    Convert Optuna trial to metrics format compatible with existing code.

    Args:
        trial: Optuna trial

    Returns:
        Metrics dictionary
    """
    metrics = {
        "config": trial.params,
        "trial_id": trial.number,
        "state": trial.state.name,
    }

    if trial.values:
        if len(trial.values) == 2:
            metrics["accuracy"] = trial.values[0]
            metrics["loss"] = trial.values[1]
        else:
            metrics["score"] = trial.values[0]

    return metrics


def optimize_with_callback(
    study: optuna.Study,
    objective: Callable,
    n_trials: int,
    callbacks: Optional[List[Callable]] = None,
) -> None:
    """
    Run optimization with custom callbacks (for UI updates).

    Args:
        study: Optuna study
        objective: Objective function
        n_trials: Number of trials to run
        callbacks: List of callback functions
    """
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
    )
