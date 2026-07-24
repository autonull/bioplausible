"""
CUDA-accelerated operations for MEP optimizers.

This module provides optimized implementations of:
- Low-rank SVD (Dion)
- Newton-Schulz iteration (Muon)
- Spectral norm estimation
"""

from .kernels import (
    batched_newton_schulz_cuda,
    dion_update_cuda,
    enforce_spectral_constraint_cuda,
    lowrank_svd_cuda,
    newton_schulz_cuda,
    spectral_norm_power_iteration_cuda,
)

__all__ = [
    "batched_newton_schulz_cuda",
    "dion_update_cuda",
    "enforce_spectral_constraint_cuda",
    "lowrank_svd_cuda",
    "newton_schulz_cuda",
    "spectral_norm_power_iteration_cuda",
]
