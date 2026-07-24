"""
Backend Detection and Configuration

Detects and configures the optimal compute backend for acceleration.
"""

import warnings

import torch

_GLOBAL_COMPILE_CHECKED = False
_GLOBAL_COMPILE_WORKS = False


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


def get_optimal_backend() -> str:
    """
    Detect best available compute backend.

    Returns:
        'cuda' | 'mps' | 'cpu'
    """
    return BackendDetector.detect_best_backend()


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
        precision = "high" if enable else "highest"
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        torch.set_float32_matmul_precision(precision)


class CupyChecker:
    """Helper class to check CuPy availability."""

    @staticmethod
    def check_availability() -> tuple[bool, str]:
        """Check if CuPy is available with proper CUDA configuration."""
        try:
            import cupy as cp

            _ = cp.zeros(10)
            return True, "CuPy available with CUDA"
        except ImportError:
            return False, "CuPy not installed. Install with: pip install cupy-cuda12x"
        except Exception as e:
            return False, f"CuPy installed but CUDA failed: {e}"


def check_cupy_available() -> tuple[bool, str]:
    """
    Check if CuPy is available with proper CUDA configuration.

    Returns:
        (available: bool, message: str)
    """
    return CupyChecker.check_availability()


TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    if not hasattr(tl, "tanh"):
        warnings.warn(
            "Triton detected but missing 'tanh'. Disabling Triton support.",
            RuntimeWarning,
        )
        TRITON_AVAILABLE = False
    else:
        TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

HAS_TRITON = TRITON_AVAILABLE


class TritonChecker:
    """Helper class to check Triton availability."""

    @staticmethod
    def check_availability() -> tuple[bool, str]:
        """Check if Triton is available for custom kernels."""
        if TRITON_AVAILABLE:
            return True, "Triton available"
        return False, "Triton not installed. Install with: pip install triton"


def check_triton_available() -> tuple[bool, str]:
    """Check if Triton is available for custom kernels."""
    return TritonChecker.check_availability()


HAS_CUPY = False
try:
    import cupy as cp

    if hasattr(cp, "cuda") and cp.cuda.is_available():
        with cp.cuda.Device(0):
            _ = cp.array([1.0])
            _ = cp.random.rand(1)
        HAS_CUPY = True
    else:
        cp = None
except ImportError, Exception:
    cp = None
    HAS_CUPY = False


__all__ = [
    "HAS_CUPY",
    "HAS_TRITON",
    "TRITON_AVAILABLE",
    "BackendDetector",
    "CupyChecker",
    "TritonChecker",
    "check_cupy_available",
    "check_triton_available",
    "enable_tf32",
    "get_optimal_backend",
]
