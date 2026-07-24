"""
Zoo Models Package

All models registered with the unified registry system.
"""

from typing import List

from bioplausible.core.registry import Domain
from bioplausible.core.registry import LocalityLevel
from bioplausible.core.registry import Registry
from bioplausible.core.registry import register_model

from . import backprop  # noqa: F401
from . import eqprop  # noqa: F401
from . import fa  # noqa: F401
from . import forward_only  # noqa: F401
from . import hebbian  # noqa: F401
from . import predictive_coding  # noqa: F401
from . import spiking  # noqa: F401
from . import target_prop  # noqa: F401

__all__: List[str] = [
    "register_model",
    "Registry",
    "Domain",
    "LocalityLevel",
]
