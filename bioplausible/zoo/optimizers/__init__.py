"""
Zoo Optimizers Package

Optimizers registered with the unified registry.
"""

from bioplausible.core.registry import Domain
from bioplausible.core.registry import LocalityLevel
from bioplausible.core.registry import register_optimizer

from . import ewc
from . import muon
from . import spectral
from . import standard

__all__ = [
    "register_optimizer",
    "Domain",
    "LocalityLevel",
    "ewc",
    "muon",
    "spectral",
    "standard",
]
