"""
Zoo Propagators Package

Learning rules / propagators registered with the unified registry.
"""

from bioplausible.core.registry import register_propagator

from . import backprop  # noqa: F401
from . import base  # noqa: F401
from . import eqprop  # noqa: F401
from . import fa  # noqa: F401
from . import forward_only  # noqa: F401
from . import hebbian  # noqa: F401
from . import mep  # noqa: F401
from . import predictive_coding  # noqa: F401
from . import spiking  # noqa: F401
from . import target_prop  # noqa: F401

__all__ = [
    "register_propagator",
]
