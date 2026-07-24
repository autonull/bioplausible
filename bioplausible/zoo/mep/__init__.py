"""
MEP: Muon Equilibrium Propagation

A biologically plausible deep learning framework using Equilibrium Propagation
with geometry-aware updates (Muon orthogonalization, Dion low-rank, spectral constraints).

Quick Start:
    from mep import smep, sdmep, muon_backprop

    # SMEP with EP
    optimizer = smep(model.parameters(), model=model, mode='ep')
    optimizer.step(x=x, target=y)

    # Muon with backprop
    optimizer = muon_backprop(model.parameters())
    loss.backward()
    optimizer.step()

See NICHES.md for optimizer selection guide.
"""

from bioplausible.core.registry import Domain
from bioplausible.core.registry import LocalityLevel
from bioplausible.core.registry import register_optimizer
from bioplausible.core.registry import register_propagator

from .optimizers import BackpropGradient
from .optimizers import CompositeOptimizer
from .optimizers import DionUpdate
from .optimizers import EnergyFunction
from .optimizers import EPGradient
from .optimizers import EPOptimizer
from .optimizers import ErrorFeedback
from .optimizers import FisherUpdate
from .optimizers import LocalEPGradient
from .optimizers import ModelInspector
from .optimizers import MuonUpdate
from .optimizers import NaturalGradient
from .optimizers import NoConstraint
from .optimizers import NoFeedback
from .optimizers import PlainUpdate
from .optimizers import Settler
from .optimizers import SpectralConstraint
from .optimizers.monitor import EPMonitor
from .optimizers.monitor import monitor_ep_training
from .presets import local_ep
from .presets import muon_backprop
from .presets import natural_ep
from .presets import sdmep
from .presets import smep
from .presets import smep_fast

__version__ = "0.3.0"
__all__ = [
    # Core optimizer
    "CompositeOptimizer",
    "EPOptimizer",  # Unified optimizer (recommended)
    # Strategy classes (for custom compositions)
    "BackpropGradient",
    "EPGradient",
    "LocalEPGradient",
    "NaturalGradient",
    "PlainUpdate",
    "MuonUpdate",
    "DionUpdate",
    "FisherUpdate",
    "NoConstraint",
    "SpectralConstraint",
    "NoFeedback",
    "ErrorFeedback",
    # Utilities
    "EnergyFunction",
    "Settler",
    "ModelInspector",
    "EPMonitor",
    "monitor_ep_training",
    # Preset factories (backward compatible)
    "smep",
    "smep_fast",
    "sdmep",
    "local_ep",
    "natural_ep",
    "muon_backprop",
    # Registry
    "register_propagator",
    "register_optimizer",
    "Domain",
    "LocalityLevel",
]

# Optional CUDA module
try:
    from . import cuda

    __all__.append("cuda")
except ImportError:
    pass
