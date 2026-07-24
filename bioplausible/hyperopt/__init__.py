"""
Hyperparameter Optimization Package for Bio-Plausible Learning Research

Powered by Optuna for multi-objective optimization.
"""

from bioplausible.execution.algorithm_constraints import (
    create_constrained_optuna_config,
    get_constrained_search_space,
)

from .eval_tiers import (
    EVALUATION_TIERS,
    EvaluationConfig,
    PatientLevel,
    estimate_total_time,
    get_evaluation_config,
    print_evaluation_summary,
)
from .optuna_bridge import (
    create_optuna_space,
    create_study,
    get_pareto_trials,
    optimize_with_callback,
    trial_to_metrics,
)
from .search_space import SEARCH_SPACES, SearchSpace, get_search_space

__version__ = "0.1.0"

# Optuna is now required
HAS_OPTUNA = True

# Evaluation tiers for patience-based optimization
# Core Optuna integration
# Search space definitions

__all__ = [
    "EVALUATION_TIERS",
    "HAS_OPTUNA",
    "SEARCH_SPACES",
    "EvaluationConfig",
    "PatientLevel",
    "SearchSpace",
    "create_constrained_optuna_config",
    "create_optuna_space",
    "create_study",
    "estimate_total_time",
    "get_constrained_search_space",
    "get_evaluation_config",
    "get_pareto_trials",
    "get_search_space",
    "optimize_with_callback",
    "print_evaluation_summary",
    "trial_to_metrics",
]
