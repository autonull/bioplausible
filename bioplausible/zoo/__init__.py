"""
Zoo Package - Unified Component Registry

All models, propagators, optimizers, sparsity methods, and other components
are registered here with rich metadata for AutoScientist composition.
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

# Import submodules to trigger registration
from bioplausible.zoo import models, optimizers, propagators, sparsity

__all__ = [
    # Registry
    "Registry",
    "ComponentCategory",
    "Domain",
    "LocalityLevel",
    "ComputeProfile",
    "ComponentMetadata",
    # Registration decorators
    "register_model",
    "register_propagator",
    "register_optimizer",
    "register_sparsity",
    "register_metric",
    "register_data_loader",
    "register_task",
    "register_callback",
    "register_domain",
    # Submodules
    "models",
    "propagators",
    "optimizers",
    "sparsity",
]
