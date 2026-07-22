"""
Zoo Sparsity Package

Sparsity methods registered with the unified registry.
"""

from bioplausible.core.registry import register_sparsity

# Import registered sparsity methods to trigger @register_sparsity decorators
from bioplausible.zoo.sparsity.registered_sparsity import (  # noqa: F401
    ActivityDrivenPruning,
    RandomPruning,
    _RegisteredTopKPruning,
)

__all__ = [
    "register_sparsity",
]
