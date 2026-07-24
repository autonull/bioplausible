"""
Core Package

Core infrastructure: Registry, CoreTrainer, Config.
"""

from bioplausible.core.registry import ComponentCategory
from bioplausible.core.registry import ComponentMetadata
from bioplausible.core.registry import ComputeProfile
from bioplausible.core.registry import Domain
from bioplausible.core.registry import LocalityLevel
from bioplausible.core.registry import Registry
from bioplausible.core.registry import register_callback
from bioplausible.core.registry import register_data_loader
from bioplausible.core.registry import register_domain
from bioplausible.core.registry import register_metric
from bioplausible.core.registry import register_model
from bioplausible.core.registry import register_optimizer
from bioplausible.core.registry import register_propagator
from bioplausible.core.registry import register_sparsity
from bioplausible.core.registry import register_task
from bioplausible.core.trainer import CoreTrainer
from bioplausible.core.trainer import TrainerConfig
from bioplausible.core.trainer import TrainingMetrics
from bioplausible.core.trainer import run_from_config

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
