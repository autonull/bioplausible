"""
Zoo Propagators Package

Learning rules / propagators registered with the unified registry.
"""

from bioplausible.core.registry import register_propagator

from . import (
    backprop,  # noqa: F401
    base,  # noqa: F401
    eqprop,  # noqa: F401
    fa,  # noqa: F401
    forward_only,  # noqa: F401
    hebbian,  # noqa: F401
    mep,  # noqa: F401
    predictive_coding,  # noqa: F401
    spiking,  # noqa: F401
    target_prop,  # noqa: F401
)

__all__ = [
    "register_propagator",
]
