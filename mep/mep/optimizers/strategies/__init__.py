"""
Strategy interfaces and implementations for MEP optimizers.

This module defines the strategy pattern components that can be composed
to create various optimizer configurations.
"""

from .base import ConstraintStrategy, FeedbackStrategy, GradientStrategy, UpdateStrategy
from .constraint import NoConstraint, SettlingSpectralPenalty, SpectralConstraint
from .feedback import ErrorFeedback, NoFeedback
from .gradient import BackpropGradient, EPGradient, LocalEPGradient, NaturalGradient
from .update import DionUpdate, FisherUpdate, MuonUpdate, PlainUpdate

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
