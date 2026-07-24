"""
Zoo Sparsity Package

Sparsity methods registered with the unified registry.
"""

from bioplausible.core.registry import register_sparsity

from . import methods  # noqa: F401  (triggers registration)

__all__ = [
    "register_sparsity",
]
