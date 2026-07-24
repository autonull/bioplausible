"""
Zoo Models Package

All models registered with the unified registry system.
"""

from typing import List

from bioplausible.core.registry import Domain, LocalityLevel, Registry, register_model

from . import (
    backprop,  # noqa: F401
    eqprop,  # noqa: F401
    fa,  # noqa: F401
    forward_only,  # noqa: F401
    hebbian,  # noqa: F401
    predictive_coding,  # noqa: F401
    spiking,  # noqa: F401
    target_prop,  # noqa: F401
)

__all__: list[str] = [
    "Domain",
    "LocalityLevel",
    "Registry",
    "register_model",
]
