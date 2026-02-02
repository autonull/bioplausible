"""
EqProp-Torch Acceleration Utilities

Provides torch.compile wrappers for 2-3x speedup and optional CuPy/Triton kernels.
Designed for portability: works on CPU, CUDA, ROCm, and Apple MPS.
"""

import warnings
from typing import Callable, Optional, Tuple

import torch


def get_optimal_backend() -> str:
    """
    Detect best available compute backend.

    Returns:
        'cuda' | 'mps' | 'cpu'
    """
    backend_detector = BackendDetector()
    return backend_detector.detect_best_backend()


class BackendDetector:
    """Helper class to detect the optimal compute backend."""

    @staticmethod
    def detect_best_backend() -> str:
        """Detect the best available compute backend."""
        backends_priority = [
            BackendDetector._get_cuda_backend,
            BackendDetector._get_mps_backend,
        ]

        for get_backend_func in backends_priority:
            backend = get_backend_func()
            if backend:
                return backend

        return "cpu"

    @staticmethod
    def _get_cuda_backend() -> str:
        """Get CUDA backend if available."""
        return "cuda" if torch.cuda.is_available() else None

    @staticmethod
    def _get_mps_backend() -> str:
        """Get MPS backend if available."""
        return "mps" if BackendDetector._is_mps_available() else None

    @staticmethod
    def _is_mps_available() -> bool:
        """Check if MPS backend is available (Apple Silicon)."""
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def enable_tf32(enable: bool = True) -> None:
    """
    Enable TensorFloat-32 (TF32) for significant speedup on Ampere+ GPUs.

    TF32 reduces precision slightly (19 bits vs 24 bits significand)
    but maintains full range, usually providing 2-3x speedup for
    matmul and convolutions with negligible accuracy loss.

    Args:
        enable: Whether to enable TF32
    """
    if torch.cuda.is_available():
        # High precision = TF32 enabled
        # Highest precision = TF32 disabled (slow)
        precision = "high" if enable else "highest"
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        torch.set_float32_matmul_precision(precision)


def check_cupy_available() -> Tuple[bool, str]:
    """
    Check if CuPy is available with proper CUDA configuration.

    Returns:
        (available: bool, message: str)
    """
    cupy_checker = CupyChecker()
    return cupy_checker.check_availability()


class CupyChecker:
    """Helper class to check CuPy availability."""

    @staticmethod
    def check_availability() -> Tuple[bool, str]:
        """Check if CuPy is available with proper CUDA configuration."""
        try:
            import cupy as cp

            # Try a simple operation to verify CUDA works
            _ = cp.zeros(10)
            return True, "CuPy available with CUDA"
        except ImportError:
            return False, "CuPy not installed. Install with: pip install cupy-cuda12x"
        except Exception as e:
            return False, f"CuPy installed but CUDA failed: {e}"


import os

# Global state for compile safety
_COMPILE_CHECKED = False
_COMPILE_WORKS = False


def _check_compile_works() -> bool:
    """Runtime check to see if torch.compile actually works."""
    global _COMPILE_CHECKED, _COMPILE_WORKS

    if _COMPILE_CHECKED:
        return _COMPILE_WORKS

    if os.environ.get("BIOPL_DISABLE_COMPILE", "0") == "1":
        _COMPILE_WORKS = False
        _COMPILE_CHECKED = True
        return False

    try:
        # Try compiling a tiny dummy function
        # CRITICAL: Must use tanh to catch missing tl.tanh in broken Triton envs
        # AND large enough tensor to trigger Triton backend (small tensors use ATen/C++)
        def dummy_fn(x):
            return torch.tanh(x * 2.0)

        compiled = torch.compile(dummy_fn, mode="reduce-overhead")
        # Must run it to trigger compilation
        _ = compiled(torch.ones(128, 128))
        _COMPILE_WORKS = True
    except Exception as e:
        warnings.warn(
            f"torch.compile check failed: {e}. Disabling compilation.", RuntimeWarning
        )
        _COMPILE_WORKS = False

    _COMPILE_CHECKED = True
    return _COMPILE_WORKS


def compile_model(
    model: torch.nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
) -> torch.nn.Module:
    """
    Wrap model with torch.compile for significant speedup.

    Works on CPU, CUDA, ROCm, and MPS without modification.
    Falls back gracefully if torch.compile is unavailable or broken.

    Args:
        model: PyTorch model to compile
        mode: Compilation mode:
            - 'default': Balanced speed and compile time
            - 'reduce-overhead': Minimize GPU kernel launch overhead
            - 'max-autotune': Maximum speed (longer compile)
        fullgraph: If True, requires entire forward to be capturable
        dynamic: Enable dynamic shapes (None = auto-detect)

    Returns:
        Compiled model (or original if compile unavailable)

    Example:
        >>> model = LoopedMLP(784, 256, 10)
        >>> model = compile_model(model, mode='reduce-overhead')
    """
    # Check PyTorch version
    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile not available (requires PyTorch 2.0+). "
            "Using uncompiled model.",
            RuntimeWarning,
        )
        return model

    if not _check_compile_works():
        return model

    # CRITICAL: If Triton is broken (missing tanh), compile might crash or produce NaNs.
    if not TRITON_AVAILABLE:
        return model

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return compiled
    except Exception as e:
        warnings.warn(
            f"torch.compile failed: {e}. Using uncompiled model.", RuntimeWarning
        )
        return model


def compile_settling_loop(settling_fn: Callable) -> Callable:
    """
    Decorator to compile the inner settling loop for maximum speed.

    Use this on the forward_step method of EqProp models:

        @compile_settling_loop
        def forward_step(self, h, x_emb):
            ...

    Args:
        settling_fn: Function to compile

    Returns:
        Compiled function
    """
    if not hasattr(torch, "compile"):
        return settling_fn

    # Respect global safety check
    if not _check_compile_works():
        return settling_fn

    # CRITICAL: If Triton is broken (missing tanh), compile might crash or produce NaNs.
    # We detected this in check_triton_available/import.
    if not TRITON_AVAILABLE:
        return settling_fn

    try:
        return torch.compile(settling_fn, mode="reduce-overhead")
    except Exception:
        return settling_fn


# =============================================================================
# Optional Triton Kernel Stub
# =============================================================================

TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    # CRITICAL CHECK for broken Triton installations
    if not hasattr(tl, "tanh"):
        warnings.warn(
            "Triton detected but missing 'tanh'. Disabling Triton support.",
            RuntimeWarning,
        )
        TRITON_AVAILABLE = False
    else:
        TRITON_AVAILABLE = True
except ImportError:
    pass


def check_triton_available() -> Tuple[bool, str]:
    """Check if Triton is available for custom kernels."""
    if TRITON_AVAILABLE:
        return True, "Triton available"
    return False, "Triton not installed. Install with: pip install triton"


# Triton kernel example (commented - requires Triton and CUDA/ROCm)
# @triton.jit
# def settling_kernel(
#     h_ptr, x_proj_ptr, W_rec_ptr,
#     hidden_dim: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr
# ):
#     """Fused settling iteration kernel."""
#     pid = tl.program_id(0)
#     # ... implementation for fused tanh(x_proj + W_rec @ h)


__all__ = [
    "get_optimal_backend",
    "check_cupy_available",
    "check_triton_available",
    "compile_model",
    "compile_settling_loop",
    "TRITON_AVAILABLE",
]
