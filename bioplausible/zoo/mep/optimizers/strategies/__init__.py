"""
Strategy interfaces and implementations for MEP optimizers.

This module defines the strategy pattern components that can be composed
to create various optimizer configurations.
"""

from .base import ConstraintStrategy
from .base import FeedbackStrategy
from .base import GradientStrategy
from .base import UpdateStrategy
from .constraint import NoConstraint
from .constraint import SettlingSpectralPenalty
from .constraint import SpectralConstraint
from .feedback import ErrorFeedback
from .feedback import NoFeedback
from .gradient import BackpropGradient
from .gradient import EPGradient
from .gradient import LocalEPGradient
from .gradient import NaturalGradient
from .update import DionUpdate
from .update import FisherUpdate
from .update import MuonUpdate
from .update import PlainUpdate

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
