"""
Bioplausible Optimizers

All optimizers in one place.

Usage:
    from bioplausible.optimizers import smep, FeedbackAlignment, Adam
    from bioplausible.optimizers import create_optimizer, list_optimizers
"""

# Base class
from .base import BioOptimizer

# Learning rules
from .learning_rules import (
    ContrastiveHebbianLearning,
    DirectFA,
    EqProp,
    FeedbackAlignment,
    FiniteNudgeEqProp,
    HolomorphicEqProp,
    LazyEqProp,
)

# MEP optimizers (from external package)
try:
    from mep.presets import local_ep, muon_backprop, natural_ep, sdmep, smep, smep_fast

    HAS_MEP = True
except ImportError:
    HAS_MEP = False
    smep = None
    smep_fast = None
    sdmep = None
    local_ep = None
    natural_ep = None
    muon_backprop = None

# Standard optimizers
from torch.optim import SGD, Adam, AdamW


# Simple optimizer registry
def _get_optimizer_registry():
    registry = {
        # Learning rules
        "feedback_alignment": FeedbackAlignment,
        "direct_fa": DirectFA,
        "eqprop": EqProp,
        "holomorphic_eqprop": HolomorphicEqProp,
        "finite_nudge": FiniteNudgeEqProp,
        "lazy_eqprop": LazyEqProp,
        "chl": ContrastiveHebbianLearning,
        # Standard
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
    }

    if HAS_MEP:
        registry.update(
            {
                "smep": smep,
                "smep_fast": smep_fast,
                "sdmep": sdmep,
                "local_ep": local_ep,
                "natural_ep": natural_ep,
                "muon_backprop": muon_backprop,
            }
        )

    return registry


OPTIMIZER_REGISTRY = _get_optimizer_registry()


def create_optimizer(model, name: str, **kwargs):
    """Create an optimizer for a model."""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    opt_class = OPTIMIZER_REGISTRY[name]

    # MEP and learning rules need model argument
    if name in [
        "smep",
        "smep_fast",
        "sdmep",
        "local_ep",
        "natural_ep",
        "muon_backprop",
        "feedback_alignment",
        "direct_fa",
        "eqprop",
        "holomorphic_eqprop",
        "finite_nudge",
        "lazy_eqprop",
        "chl",
    ]:
        return opt_class(model.parameters(), model=model, **kwargs)
    else:
        return opt_class(model.parameters(), **kwargs)


def list_optimizers():
    """List available optimizers."""
    return list(OPTIMIZER_REGISTRY.keys())


__all__ = [
    # Base
    "BioOptimizer",
    # Learning rules
    "FeedbackAlignment",
    "DirectFA",
    "EqProp",
    "HolomorphicEqProp",
    "FiniteNudgeEqProp",
    "LazyEqProp",
    "ContrastiveHebbianLearning",
    # MEP
    "smep",
    "smep_fast",
    "sdmep",
    "local_ep",
    "natural_ep",
    "muon_backprop",
    # Standard
    "SGD",
    "Adam",
    "AdamW",
    # Factory
    "create_optimizer",
    "list_optimizers",
]
