"""
MEP Optimizers: Refactored strategy pattern implementation.

This module provides a composable optimizer framework for:
- Equilibrium Propagation (EP)
- Muon orthogonalization (Newton-Schulz)
- Dion low-rank updates (SVD)
- Spectral norm constraints
- Natural gradient descent

Quick Start:
    from mep.optimizers import smep, sdmep, muon_backprop
    
    # SMEP with EP
    optimizer = smep(model.parameters(), model=model, mode='ep')
    optimizer.step(x=x, target=y)
    
    # Muon with backprop (drop-in SGD replacement)
    optimizer = muon_backprop(model.parameters())
    loss.backward()
    optimizer.step()
"""

from .composite import CompositeOptimizer
from .strategies import (
    # Interfaces
    GradientStrategy,
    UpdateStrategy,
    ConstraintStrategy,
    FeedbackStrategy,
    # Implementations
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
    SettlingSpectralPenalty,
    NoFeedback,
    ErrorFeedback,
)
from .energy import EnergyFunction
from .settling import Settler
from .inspector import ModelInspector
from .o1_memory import (
    manual_energy_compute,
    settle_manual,
    energy_from_states,
    O1MemoryEP,
)
from .o1_memory_v2 import (
    analytic_state_gradients,
    settle_manual_o1,
    manual_energy_compute_o1,
    energy_from_states_minimal,
    O1MemoryEPv2,
)
from .ewc import (
    EWCRegularizer,
    EPOptimizerWithEWC,
    TaskMemory,
)
from .ep_optimizer import (
    EPOptimizer,
    EPConfig,
    EWCState,
    smep,
    smep_fast,
    muon_backprop,
)

__all__ = [
    # Core optimizer
    "CompositeOptimizer",
    # Strategy interfaces
    "GradientStrategy",
    "UpdateStrategy",
    "ConstraintStrategy",
    "FeedbackStrategy",
    # Gradient strategies
    "BackpropGradient",
    "EPGradient",
    "LocalEPGradient",
    "NaturalGradient",
    # Update strategies
    "PlainUpdate",
    "MuonUpdate",
    "DionUpdate",
    "FisherUpdate",
    # Constraint strategies
    "NoConstraint",
    "SpectralConstraint",
    "SettlingSpectralPenalty",
    # Feedback strategies
    "NoFeedback",
    "ErrorFeedback",
    # Utilities
    "EnergyFunction",
    "Settler",
    "ModelInspector",
    # Unified EP optimizer (recommended)
    "EPOptimizer",
    "EPConfig",
    "EWCState",
    # Legacy presets (backward compatible)
    "smep",
    "smep_fast",
    "muon_backprop",
    # O(1) memory prototype v1 (legacy)
    "manual_energy_compute",
    "settle_manual",
    "energy_from_states",
    "O1MemoryEP",
    # O(1) memory prototype v2 (legacy)
    "analytic_state_gradients",
    "settle_manual_o1",
    "manual_energy_compute_o1",
    "energy_from_states_minimal",
    "O1MemoryEPv2",
    # EWC for continual learning (legacy)
    "EWCRegularizer",
    "EPOptimizerWithEWC",
    "TaskMemory",
]
