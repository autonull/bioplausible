"""
Core Package

Core infrastructure: Registry, CoreTrainer, Config.
"""

from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
    register_callback,
    register_data_loader,
    register_domain,
    register_metric,
    register_model,
    register_optimizer,
    register_propagator,
    register_sparsity,
    register_task,
)
from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
    TrainingMetrics,
    run_from_config,
)

__all__ = [
    # Registry
    "Registry",
    "ComponentCategory",
    "Domain",
    "LocalityLevel",
    "ComputeProfile",
    "ComponentMetadata",
    "register_model",
    "register_propagator",
    "register_optimizer",
    "register_sparsity",
    "register_metric",
    "register_data_loader",
    "register_task",
    "register_callback",
    "register_domain",
    # Trainer
    "CoreTrainer",
    "TrainerConfig",
    "TrainingMetrics",
    "run_from_config",
]
