"""
Hyperparameter Optimization Package for Bio-Plausible Learning Research

Powered by Optuna for multi-objective optimization.
"""

from .search_space import SEARCH_SPACES, SearchSpace, get_search_space
from .optuna_bridge import (create_optuna_space, create_study,
                            get_pareto_trials, optimize_with_callback,
                            trial_to_metrics)
from .eval_tiers import (EVALUATION_TIERS, EvaluationConfig, PatientLevel,
                         estimate_total_time, get_evaluation_config,
                         print_evaluation_summary)
from bioplausible.scientist.algorithm_constraints import (
    get_constrained_search_space,
    create_constrained_optuna_config
)

__version__ = "0.1.0"

# Optuna is now required
HAS_OPTUNA = True

# Evaluation tiers for patience-based optimization
# Core Optuna integration
# Search space definitions

__all__ = [
    "create_optuna_space",
    "create_study",
    "get_pareto_trials",
    "optimize_with_callback",
    "trial_to_metrics",
    "SearchSpace",
    "get_search_space",
    "SEARCH_SPACES",
    "HAS_OPTUNA",
    "PatientLevel",
    "EvaluationConfig",
    "EVALUATION_TIERS",
    "get_evaluation_config",
    "estimate_total_time",
    "print_evaluation_summary",
    "get_constrained_search_space",
    "create_constrained_optuna_config",
]
