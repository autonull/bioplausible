"""
Strategy interfaces and implementations for MEP optimizers.

This module defines the strategy pattern components that can be composed
to create various optimizer configurations.
"""

from .base import (
    GradientStrategy,
    UpdateStrategy,
    ConstraintStrategy,
    FeedbackStrategy,
)
from .gradient import (
    BackpropGradient,
    EPGradient,
    LocalEPGradient,
    NaturalGradient,
)
from .update import (
    PlainUpdate,
    MuonUpdate,
    DionUpdate,
    FisherUpdate,
)
from .constraint import (
    NoConstraint,
    SpectralConstraint,
    SettlingSpectralPenalty,
)
from .feedback import (
    NoFeedback,
    ErrorFeedback,
)

__all__ = [
    # Interfaces
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
]
