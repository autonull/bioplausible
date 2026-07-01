"""
Configuration schemas and defaults for Bioplausible experiments.

OmegaConf-based structured configs with Pydantic validation.
"""

from bioplausible.config.defaults import DEFAULT_CONFIGS
from bioplausible.config.schema import (DatasetConfig, DomainConfig,
                                        ExperimentConfig, LightningConfig,
                                        ModelConfig, OptimizerConfig,
                                        PropagatorConfig, ScientistConfig,
                                        SparsityConfig, TrainingConfig,
                                        get_default_config, validate_config)
# Backward compatibility: GLOBAL_CONFIG and legacy config exports
from bioplausible.config_legacy import (COMPILE_CONFIG, DATASET_CONFIG,
                                        GLOBAL_CONFIG, KERNEL_CONFIG,
                                        MODEL_PRESETS, TRAINING_DEFAULTS)
from bioplausible.config_legacy import \
    TrainerConfig as LegacyTrainerConfig  # noqa: F401
from bioplausible.config_legacy import get_model_config

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
]
