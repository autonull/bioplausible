from dataclasses import dataclass
from typing import Any, Dict, Optional
from bioplausible.hyperopt import PatientLevel

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
