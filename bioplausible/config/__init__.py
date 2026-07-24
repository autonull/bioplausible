"""
Configuration schemas and defaults for Bioplausible experiments.

OmegaConf-based structured configs with Pydantic validation.
"""

import os
import pathlib
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field, ValidationError

from bioplausible.config.defaults import DEFAULT_CONFIGS
from bioplausible.config.schema import (
    DatasetConfig,
    DomainConfig,
    ExperimentConfig,
    LightningConfig,
    ModelConfig,
    OptimizerConfig,
    PropagatorConfig,
    ScientistConfig,
    SparsityConfig,
    TrainingConfig,
    get_default_config,
    validate_config,
)

# Backward compatibility: GLOBAL_CONFIG and legacy config exports
from bioplausible.config_legacy import (
    COMPILE_CONFIG,
    DATASET_CONFIG,
    GLOBAL_CONFIG,
    KERNEL_CONFIG,
    MODEL_PRESETS,
    TRAINING_DEFAULTS,
    get_model_config,
)
from bioplausible.config_legacy import (
    TrainerConfig as LegacyTrainerConfig,
)

# ──────────────────────────────────────────────
# Merged from config_loader.py
# ──────────────────────────────────────────────


class ExperimentSchema(BaseModel):
    """Schema for validating experiment configurations."""

    model: str = Field(..., description="Name of the model (e.g., LoopedMLP)")
    task: str = Field(default="mnist", description="Task name")
    hyperparams: dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    training: dict[str, Any] = Field(
        default_factory=dict, description="Training settings (lr, epochs)"
    )
    description: Optional[str] = None


def load_config(path: str) -> dict[str, Any]:
    """Load and validate experiment configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the validated configuration.
    """
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with pathlib.Path(path).open() as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")
    try:
        validated_config = ExperimentSchema(**raw_config)
        return validated_config.model_dump()
    except ValidationError as e:
        raise ValueError(f"Invalid configuration format: {e}")


__all__ = [
    # Legacy exports
    "GLOBAL_CONFIG",
    "TRAINING_DEFAULTS",
    "MODEL_PRESETS",
    "DATASET_CONFIG",
    "COMPILE_CONFIG",
    "KERNEL_CONFIG",
    "LegacyTrainerConfig",
    "get_model_config",
    # New schema exports
    "ExperimentConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PropagatorConfig",
    "SparsityConfig",
    "TrainingConfig",
    "DatasetConfig",
    "LightningConfig",
    "DomainConfig",
    "ScientistConfig",
    "get_default_config",
    "validate_config",
    "DEFAULT_CONFIGS",
    # Merged from config_loader.py
    "ExperimentSchema",
    "load_config",
]
