"""
EqProp-Torch Acceleration Utilities

This module re-exports from bioplausible.acceleration for backward compatibility.
All acceleration utilities have been consolidated into the acceleration package.

For new code, use:
    from bioplausible.acceleration import (
        get_optimal_backend,
        compile_model,
        HAS_CUPY,
        TRITON_AVAILABLE,
    )
"""

from bioplausible.acceleration import HAS_CUPY
from bioplausible.acceleration import HAS_TRITON
from bioplausible.acceleration import TRITON_AVAILABLE
from bioplausible.acceleration import BackendDetector
from bioplausible.acceleration import CupyChecker
from bioplausible.acceleration import TritonChecker
from bioplausible.acceleration import check_cupy_available
from bioplausible.acceleration import check_triton_available
from bioplausible.acceleration import compile_model
from bioplausible.acceleration import compile_settling_loop
from bioplausible.acceleration import enable_tf32
from bioplausible.acceleration import get_optimal_backend

__all__ = [
    "get_optimal_backend",
    "check_cupy_available",
    "check_triton_available",
    "compile_model",
    "compile_settling_loop",
    "TRITON_AVAILABLE",
    "HAS_CUPY",
    "HAS_TRITON",
    "enable_tf32",
    "BackendDetector",
    "CupyChecker",
    "TritonChecker",
]
