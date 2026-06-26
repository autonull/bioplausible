from dataclasses import dataclass
from typing import Any, Dict, Optional

from bioplausible.hyperopt import PatientLevel


@dataclass
class ExperimentTask:
    """
    Represents a single planned experiment task.

    Attributes:
        model_name: The name of the model to run (e.g., 'mlp', 'transformer').
        task_name: The dataset/task to run on (e.g., 'mnist', 'cifar10').
        tier: The rigorousness tier (SMOKE, SHALLOW, STANDARD, DEEP, CROSS_VAL).
        study_name: The Optuna study name for this experiment.
        priority: Priority score for scheduling (higher is better).
        fixed_config: Optional dictionary for fixed hyperparameters (Verification/Ablation).
        verification_of_trial_id: ID of the original trial being verified.
        fold_index: Fold index (0-4) for Cross-Validation.
        last_run_timestamp: Timestamp of the last execution.
        is_robustness_check: Whether this is a robustness analysis task.
        is_ablation: Whether this is an ablation study task.
        ablation_param: The parameter being ablated.
        is_transfer: Whether this is a transfer learning task.
        transfer_from_trial: ID of the source trial for transfer learning.
        is_continual: Whether this is a continual learning task.
        continual_step: Step number in the continual learning sequence.
        constraints: Search space constraints based on prior knowledge.
    """

    model_name: str
    task_name: str
    tier: PatientLevel
    study_name: str
    priority: float
    fixed_config: Optional[Dict[str, Any]] = None
    verification_of_trial_id: Optional[int] = None
    fold_index: Optional[int] = None
    last_run_timestamp: Optional[str] = None
    is_robustness_check: bool = False
    is_ablation: bool = False
    ablation_param: Optional[str] = None
    is_transfer: bool = False
    transfer_from_trial: Optional[int] = None
    is_continual: bool = False
    continual_step: int = 0
    constraints: Optional[Dict[str, Any]] = None
    is_evolve: bool = False
    evolve_problem: Optional[str] = None
