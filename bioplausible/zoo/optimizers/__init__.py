"""
Zoo Optimizers Package

Optimizers registered with the unified registry.
"""

from bioplausible.core.registry import Domain, LocalityLevel, register_optimizer

from . import ewc, muon, spectral, standard

__all__ = [
    "Domain",
    "LocalityLevel",
    "ewc",
    "muon",
    "register_optimizer",
    "spectral",
    "standard",
]
