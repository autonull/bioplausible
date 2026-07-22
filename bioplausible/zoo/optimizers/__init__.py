"""
Zoo Optimizers Package

Optimizers registered with the unified registry.
"""

from bioplausible.core.registry import register_optimizer

# Import registered optimizers to trigger @register_optimizer decorators
from bioplausible.zoo.optimizers.registered_optimizers import (  # noqa: F401
    _RegisteredAdam,
    _RegisteredAdamW,
    _RegisteredSGD,
)

__all__ = [
    "register_optimizer",
]
