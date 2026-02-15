"""
NumPy/CuPy Kernel for Equilibrium Propagation

Pure NumPy/CuPy implementation without PyTorch autograd.
Provides O(1) memory training via contrastive Hebbian updates.

This module re-exports from bioplausible.kernel for backward compatibility.
"""

from bioplausible.kernel import (
    EqPropKernel,
    EqPropKernelBPTT,
    cross_entropy,
    get_backend,
    softmax,
    spectral_normalize,
    to_numpy,
)

__all__ = [
    "EqPropKernel",
    "EqPropKernelBPTT",
    "get_backend",
    "to_numpy",
    "softmax",
    "cross_entropy",
    "spectral_normalize",
]
