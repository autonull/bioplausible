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

from .optimizers import (
    CompositeOptimizer,
    # Strategies
    BackpropGradient,
    EPGradient,
    LocalEPGradient,
    NaturalGradient,
    PlainUpdate,
    MuonUpdate,
    DionUpdate,
    FisherUpdate,
    NoConstraint,
    SpectralConstraint,
    NoFeedback,
    ErrorFeedback,
    EnergyFunction,
    Settler,
    ModelInspector,
    # Unified optimizer (recommended)
    EPOptimizer,
)
from .optimizers.monitor import EPMonitor, monitor_ep_training
from .presets import smep, smep_fast, sdmep, local_ep, natural_ep, muon_backprop

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
]

# Optional CUDA module
try:
    from . import cuda
    __all__.append("cuda")
except ImportError:
    pass
