"""
Hyperparameter Optimization Package for Bio-Plausible Learning Research

Powered by Optuna for multi-objective optimization.
"""

from bioplausible.execution.algorithm_constraints import (
    create_constrained_optuna_config,
)
from bioplausible.execution.algorithm_constraints import get_constrained_search_space

from .eval_tiers import EVALUATION_TIERS
from .eval_tiers import EvaluationConfig
from .eval_tiers import PatientLevel
from .eval_tiers import estimate_total_time
from .eval_tiers import get_evaluation_config
from .eval_tiers import print_evaluation_summary
from .optuna_bridge import create_optuna_space
from .optuna_bridge import create_study
from .optuna_bridge import get_pareto_trials
from .optuna_bridge import optimize_with_callback
from .optuna_bridge import trial_to_metrics
from .search_space import SEARCH_SPACES
from .search_space import SearchSpace
from .search_space import get_search_space

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
