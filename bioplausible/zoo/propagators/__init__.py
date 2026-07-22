"""
Zoo Propagators Package

Learning rules / propagators registered with the unified registry.
"""

from bioplausible.core.registry import register_propagator

# Import registered propagators to trigger @register_propagator decorators
from bioplausible.zoo.propagators.registered_propagators import (  # noqa: F401
    _RegisteredAdaptiveFA,
    _RegisteredCHL,
    _RegisteredContrastiveFA,
    _RegisteredDirectFA,
    _RegisteredEqProp,
    _RegisteredFeedbackAlignment,
    _RegisteredFiniteNudgeEqProp,
    _RegisteredHolomorphicEqProp,
    _RegisteredLazyEqProp,
    _RegisteredStochasticFA,
)

__all__ = [
    "register_propagator",
]
