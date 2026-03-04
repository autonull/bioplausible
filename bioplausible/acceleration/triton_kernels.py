"""
Triton Kernels for EqProp Acceleration

Provides fused kernels for Equilibrium Propagation dynamics to maximize
GPU throughput by reducing memory bandwidth usage.

This module re-exports from bioplausible.models.triton_kernel for
centralized access through the acceleration module.
"""

from bioplausible.models.triton_kernel import (HAS_CUPY, HAS_TRITON,
                                               TritonEqPropOps)

__all__ = [
    "TritonEqPropOps",
    "HAS_TRITON",
    "HAS_CUPY",
]
