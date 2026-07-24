"""
Configuration schemas and defaults for Bioplausible experiments.

OmegaConf-based structured configs with Pydantic validation.
"""

import os
from typing import Any
from typing import Dict

import yaml
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from bioplausible.config.defaults import DEFAULT_CONFIGS
from bioplausible.config.schema import DatasetConfig
from bioplausible.config.schema import DomainConfig
from bioplausible.config.schema import ExperimentConfig
from bioplausible.config.schema import LightningConfig
from bioplausible.config.schema import ModelConfig
from bioplausible.config.schema import OptimizerConfig
from bioplausible.config.schema import PropagatorConfig
from bioplausible.config.schema import ScientistConfig
from bioplausible.config.schema import SparsityConfig
from bioplausible.config.schema import TrainingConfig
from bioplausible.config.schema import get_default_config
from bioplausible.config.schema import validate_config

# Backward compatibility: GLOBAL_CONFIG and legacy config exports
from bioplausible.config_legacy import COMPILE_CONFIG
from bioplausible.config_legacy import DATASET_CONFIG
from bioplausible.config_legacy import GLOBAL_CONFIG
from bioplausible.config_legacy import KERNEL_CONFIG
from bioplausible.config_legacy import MODEL_PRESETS
from bioplausible.config_legacy import TRAINING_DEFAULTS
from bioplausible.config_legacy import (
    TrainerConfig as LegacyTrainerConfig,
)  # noqa: F401
from bioplausible.config_legacy import get_model_config

# ──────────────────────────────────────────────
# Merged from config_loader.py
# ──────────────────────────────────────────────


class ExperimentSchema(BaseModel):
    """Schema for validating experiment configurations."""

    model: str = Field(..., description="Name of the model (e.g., LoopedMLP)")
    task: str = Field(default="mnist", description="Task name")
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    training: Dict[str, Any] = Field(
        default_factory=dict, description="Training settings (lr, epochs)"
    )
    description: Optional[str] = None


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate experiment configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the validated configuration.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
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
