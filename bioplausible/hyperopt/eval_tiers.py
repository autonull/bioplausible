"""
Evaluation Tiers for Fair Algorithm Comparison

Provides patience/depth control for hyperparameter optimization that scales
epochs, model sizes, and trial counts based on available compute time.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class PatientLevel(Enum):
    """
    Patience levels for hyperparameter optimization.

    SMOKE: Ultra-fast smoke test (1-2 min per trial)
    SHALLOW: Quick exploration (5-10 min per trial)
    STANDARD: Balanced evaluation (30-60 min per trial)
    DEEP: Thorough overnight run (2-4 hours per trial)
    CROSS_VAL: K-Fold Cross Validation for rigorous validation
    """

    SMOKE = "smoke"
    SHALLOW = "shallow"
    STANDARD = "standard"
    DEEP = "deep"
    CROSS_VAL = "cross_val"


@dataclass
class EvaluationConfig:
    """Configuration for algorithm evaluation at a given patience level."""

    # Training parameters
    epochs: int
    max_hidden_dim: int
    max_layers: int
    batch_size: int

    # Hyperopt parameters
    n_trials: int
    n_startup_trials: int  # For TPE warmup

    # Data parameters
    train_samples: Optional[int]  # None = full dataset
    val_samples: Optional[int]

    # Compute parameters
    max_time_per_trial_minutes: float
    use_pruning: bool


# Define standard evaluation tiers
EVALUATION_TIERS: Dict[PatientLevel, EvaluationConfig] = {
    PatientLevel.SMOKE: EvaluationConfig(
        epochs=3,  # Updated to 3
        max_hidden_dim=64,
        max_layers=2,
        batch_size=256,
        n_trials=3,
        n_startup_trials=1,
        train_samples=500,  # Tiny subset
        val_samples=100,
        max_time_per_trial_minutes=1.0,
        use_pruning=True,
    ),
    PatientLevel.SHALLOW: EvaluationConfig(
        epochs=7,  # Updated to 7
        max_hidden_dim=128,
        max_layers=4,
        batch_size=128,
        n_trials=10,
        n_startup_trials=3,
        train_samples=2000,  # Small subset
        val_samples=500,
        max_time_per_trial_minutes=2.0,
        use_pruning=True,
    ),
    PatientLevel.STANDARD: EvaluationConfig(
        epochs=15,  # Updated to 15 (adaptive epoch budget)
        max_hidden_dim=256,
        max_layers=10,
        batch_size=64,
        n_trials=50,
        n_startup_trials=10,
        train_samples=None,  # Full dataset
        val_samples=None,
        max_time_per_trial_minutes=10.0,
        use_pruning=True,
    ),
    PatientLevel.DEEP: EvaluationConfig(
        epochs=30,  # Updated to 30 (adaptive epoch budget)
        max_hidden_dim=512,
        max_layers=30,
        batch_size=32,
        n_trials=200,
        n_startup_trials=20,
        train_samples=None,  # Full dataset
        val_samples=None,
        max_time_per_trial_minutes=60.0,
        use_pruning=False,  # Let trials complete
    ),
    PatientLevel.CROSS_VAL: EvaluationConfig(
        epochs=30,  # Same as standard
        max_hidden_dim=256,
        max_layers=10,
        batch_size=64,
        n_trials=5,  # 5 Folds
        n_startup_trials=0,
        train_samples=None,
        val_samples=None,
        max_time_per_trial_minutes=20.0,
        use_pruning=False,
    ),
}


def get_evaluation_config(
    patience: PatientLevel = PatientLevel.SHALLOW,
    model_family: Optional[str] = None,  # Kept for API compatibility but not used
) -> EvaluationConfig:
    """
    Get evaluation configuration for a given patience level.

    Args:
        patience: Patience level (smoke, shallow, standard, deep)
        model_family: Ignored - all models get same config for fair comparison

    Returns:
        EvaluationConfig with appropriate parameters

    Example:
        >>> config = get_evaluation_config(PatientLevel.SMOKE)
        >>> print(f"Epochs: {config.epochs}, Trials: {config.n_trials}")
        Epochs: 3, Trials: 5
    """
    return EVALUATION_TIERS[patience]


def estimate_total_time(
    patience: PatientLevel,
    n_models: int = 1,
    model_family: Optional[str] = None,
) -> Dict[str, float]:
    """
    Estimate total optimization time for comparison.

    Args:
        patience: Patience level
        n_models: Number of models to compare
        model_family: Optional model family

    Returns:
        Dictionary with time estimates in various units

    Example:
        >>> time_est = estimate_total_time(PatientLevel.SHALLOW, n_models=3)
        >>> print(f"Total: {time_est['hours']:.1f} hours")
        Total: 1.0 hours
    """
    config = get_evaluation_config(patience, model_family)

    # Estimate time per trial (assume 50% pruning effectiveness)
    if config.use_pruning:
        avg_time = config.max_time_per_trial_minutes * 0.5
    else:
        avg_time = config.max_time_per_trial_minutes

    total_minutes = avg_time * config.n_trials * n_models

    return {
        "minutes": total_minutes,
        "hours": total_minutes / 60,
        "trials_per_model": config.n_trials,
        "estimated_completion": (
            f"{total_minutes/60:.1f}h"
            if total_minutes > 60
            else f"{total_minutes:.0f}min"
        ),
    }


def print_evaluation_summary(patience: PatientLevel, n_models: int = 1):
    """
    Print a human-readable summary of the evaluation parameters.

    Example:
        >>> print_evaluation_summary(PatientLevel.SHALLOW, n_models=3)

        Evaluation Tier: SHALLOW (quick exploration)
        ==========================================
        Training: 10 epochs, up to 128 hidden units, max 6 layers
        Dataset: 5000 train / 1000 val samples
        Optimization: 20 trials with TPE sampler
        Estimated time: 1.0h (3 models)
    """
    config = get_evaluation_config(patience)
    time_est = estimate_total_time(patience, n_models)

    tier_descriptions = {
        PatientLevel.SMOKE: "ultra-fast smoke test",
        PatientLevel.SHALLOW: "quick exploration",
        PatientLevel.STANDARD: "balanced evaluation",
        PatientLevel.DEEP: "thorough overnight run",
        PatientLevel.CROSS_VAL: "rigorous cross-validation",
    }

    print(
        f"\nEvaluation Tier: {patience.value.upper()} ({tier_descriptions[patience]})"
    )
    print("=" * 60)
    print(
        f"Training: {config.epochs} epochs, up to {config.max_hidden_dim} hidden"
        f" units, max {config.max_layers} layers"
    )

    if config.train_samples:
        print(
            f"Dataset: {config.train_samples} train / {config.val_samples} val samples"
        )
    else:
        print("Dataset: Full dataset")

    print(f"Optimization: {config.n_trials} trials with TPE sampler")
    print(f"Pruning: {'Enabled' if config.use_pruning else 'Disabled'}")
    print(f"Estimated time: {time_est['estimated_completion']}", end="")
    if n_models > 1:
        print(f" ({n_models} models)")
    else:
        print()
    print()
