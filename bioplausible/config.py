"""
Configuration defaults for EqProp-Torch

Centralized configuration for common hyperparameters and settings.
"""

from dataclasses import dataclass
from typing import Any, Dict

# Default training hyperparameters
TRAINING_DEFAULTS = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "weight_decay": 0.0,
    "use_spectral_norm": True,
    "max_steps": 30,
}

# Model presets for quick experiments
MODEL_PRESETS = {
    "mnist_tiny": {
        "hidden_dim": 64,
        "max_steps": 20,
    },
    "mnist_small": {
        "hidden_dim": 128,
        "max_steps": 30,
    },
    "mnist_medium": {
        "hidden_dim": 256,
        "max_steps": 30,
    },
    "mnist_large": {
        "hidden_dim": 512,
        "max_steps": 40,
    },
    "cifar_conv_small": {
        "hidden_channels": 32,
        "max_steps": 25,
    },
    "cifar_conv_medium": {
        "hidden_channels": 64,
        "max_steps": 25,
    },
    "lm_tiny": {
        "hidden_dim": 128,
        "num_layers": 2,
        "eq_steps": 10,
    },
    "lm_small": {
        "hidden_dim": 256,
        "num_layers": 4,
        "eq_steps": 15,
    },
    "lm_medium": {
        "hidden_dim": 384,
        "num_layers": 6,
        "eq_steps": 15,
    },
}

# Dataset configurations
DATASET_CONFIG = {
    "mnist": {
        "input_dim": 784,
        "output_dim": 10,
        "input_channels": 1,
        "image_size": (28, 28),
    },
    "fashion_mnist": {
        "input_dim": 784,
        "output_dim": 10,
        "input_channels": 1,
        "image_size": (28, 28),
    },
    "cifar10": {
        "input_dim": 3072,
        "output_dim": 10,
        "input_channels": 3,
        "image_size": (32, 32),
    },
    "kmnist": {
        "input_dim": 784,
        "output_dim": 10,
        "input_channels": 1,
        "image_size": (28, 28),
    },
}

# Compilation settings
COMPILE_CONFIG = {
    "default_mode": "reduce-overhead",
    "modes": ["default", "reduce-overhead", "max-autotune"],
    "fallback_on_error": True,
}

# Kernel settings
KERNEL_CONFIG = {
    "max_steps": 10,
    "beta": 0.22,
    "gamma": 0.5,
    "adaptive_epsilon": True,
}


@dataclass
class TrainerConfig:
    """Global configuration for the trainer."""

    epochs: int = 3  # Centralized default for all experiments (baseline & trials)
    quick_mode: bool = True
    max_trial_time: float = (
        60.0  # Total trial budget in seconds (used to derive per-epoch limit)
    )
    task: str = "shakespeare"
    device: str = "auto"
    seed: int = 42

    # Advanced flags
    use_compile: bool = True
    use_kernel: str = "auto"  # "auto", "explicit", "none"


# Global instance
GLOBAL_CONFIG = TrainerConfig()


def get_model_config(preset_name: str, **overrides) -> Dict[str, Any]:
    """
    Get model configuration from preset with optional overrides.

    Args:
        preset_name: Name of preset (e.g., 'mnist_small')
        **overrides: Override specific values

    Returns:
        Configuration dict

    Example:
        >>> config = get_model_config('mnist_small', hidden_dim=256)
    """
    _validate_preset_name(preset_name)

    config = MODEL_PRESETS[preset_name].copy()
    config.update(overrides)
    return config


def _validate_preset_name(preset_name: str) -> None:
    """Validate that the preset name exists."""
    if preset_name not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(MODEL_PRESETS.keys())}"
        )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset configuration.

    Args:
        dataset_name: Name of dataset (e.g., 'mnist')

    Returns:
        Configuration dict with input_dim, output_dim, etc.
    """
    dataset_name = dataset_name.lower().replace("-", "_")

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIG.keys())}"
        )

    return DATASET_CONFIG[dataset_name].copy()


__all__ = [
    "TRAINING_DEFAULTS",
    "MODEL_PRESETS",
    "DATASET_CONFIG",
    "COMPILE_CONFIG",
    "KERNEL_CONFIG",
    "GLOBAL_CONFIG",
    "TrainerConfig",
    "get_model_config",
    "get_dataset_info",
]
