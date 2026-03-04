"""
Acceleration Module for Bioplausible

Provides multiple acceleration backends for Equilibrium Propagation:

Backends (in order of priority for speed):
    1. Triton Kernels: Custom GPU kernels for fused operations (fastest)
    2. CuPy: NumPy-compatible GPU arrays via CUDA
    3. torch.compile: PyTorch 2.0+ JIT compilation
    4. Pure PyTorch: Standard autograd (fallback)
    5. Pure NumPy: CPU-only kernel (portability)

Usage:
    from bioplausible.acceleration import (
        get_optimal_backend,
        compile_model,
        EqPropKernel,
        TritonEqPropOps,
        HAS_CUPY,
        HAS_TRITON,
    )

    # Check available backends
    print(f"CuPy: {HAS_CUPY}, Triton: {HAS_TRITON}")

    # Use optimal backend
    device = get_optimal_backend()

    # Compile model for speed
    model = compile_model(model, mode='reduce-overhead')
"""

from typing import Any, Optional, Tuple

import numpy as np

from bioplausible.acceleration.backends import (HAS_CUPY, HAS_TRITON,
                                                TRITON_AVAILABLE,
                                                BackendDetector, CupyChecker,
                                                TritonChecker,
                                                check_cupy_available,
                                                check_triton_available,
                                                enable_tf32,
                                                get_optimal_backend)
from bioplausible.acceleration.compile import (compile_model,
                                               compile_settling_loop)

EqPropKernel = None
EqPropKernelBPTT = None
TritonEqPropOps = None
HAS_TRITON_OPS = False


def _get_kernel_classes():
    """Lazily import kernel classes to avoid circular imports."""
    global EqPropKernel, EqPropKernelBPTT
    if EqPropKernel is None:
        from bioplausible.kernel import EqPropKernel as _EqPropKernel
        from bioplausible.kernel import EqPropKernelBPTT as _EqPropKernelBPTT

        EqPropKernel = _EqPropKernel
        EqPropKernelBPTT = _EqPropKernelBPTT
    return EqPropKernel, EqPropKernelBPTT


def _get_triton_ops():
    """Lazily import Triton ops to avoid import errors on systems without Triton."""
    global TritonEqPropOps, HAS_TRITON_OPS
    if TritonEqPropOps is None:
        try:
            from bioplausible.models.triton_kernel import \
                TritonEqPropOps as _TritonEqPropOps

            TritonEqPropOps = _TritonEqPropOps
            HAS_TRITON_OPS = True
        except ImportError:
            HAS_TRITON_OPS = False
    return TritonEqPropOps


def get_backend(use_gpu: bool) -> Any:
    """Return appropriate array library (CuPy or NumPy)."""
    if use_gpu and HAS_CUPY:
        import cupy as cp

        return cp
    return np


def to_numpy(arr: Any) -> np.ndarray:
    """Convert array to NumPy (handles both NumPy and CuPy arrays)."""
    if HAS_CUPY:
        try:
            import cupy as cp

            if hasattr(arr, "__class__") and arr.__class__.__module__.startswith(
                "cupy"
            ):
                return cp.asnumpy(arr)
        except Exception:
            pass
    return arr


def softmax(x: np.ndarray, xp: Any = None) -> np.ndarray:
    """Stable softmax."""
    if xp is None:
        xp = np
    x_max = xp.max(x, axis=-1, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray, xp: Any = None) -> float:
    """Cross-entropy loss from logits."""
    if xp is None:
        xp = np
    batch_size = logits.shape[0]
    probs = softmax(logits, xp)
    probs = xp.clip(probs, 1e-10, 1.0)
    log_probs = xp.log(probs)
    loss = -xp.sum(log_probs[xp.arange(batch_size), targets]) / batch_size
    return float(loss)


def spectral_normalize(
    W: np.ndarray,
    num_iters: int = 5,
    u: Optional[np.ndarray] = None,
    xp: Any = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Power iteration spectral normalization."""
    if xp is None:
        xp = np
    out_dim, in_dim = W.shape

    if u is None:
        u = xp.random.randn(out_dim).astype(W.dtype)
    u = u / xp.linalg.norm(u)

    for _ in range(num_iters):
        v = W.T @ u
        v = v / (xp.linalg.norm(v) + 1e-12)
        u = W @ v
        u = u / (xp.linalg.norm(u) + 1e-12)

    sigma = float(u @ W @ v)
    W_normalized = W / (sigma + 1e-12)

    return W_normalized, u, sigma


__all__ = [
    "HAS_CUPY",
    "HAS_TRITON",
    "HAS_TRITON_OPS",
    "TRITON_AVAILABLE",
    "get_optimal_backend",
    "check_cupy_available",
    "check_triton_available",
    "enable_tf32",
    "compile_model",
    "compile_settling_loop",
    "EqPropKernel",
    "EqPropKernelBPTT",
    "TritonEqPropOps",
    "get_backend",
    "to_numpy",
    "softmax",
    "cross_entropy",
    "spectral_normalize",
    "BackendDetector",
    "CupyChecker",
    "TritonChecker",
    "_get_kernel_classes",
    "_get_triton_ops",
]
