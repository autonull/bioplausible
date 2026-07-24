"""
Model Compilation Utilities

Provides torch.compile wrappers for 2-3x speedup.
"""

import os
import warnings
from collections.abc import Callable

import torch

from bioplausible.acceleration.backends import TRITON_AVAILABLE

_GLOBAL_COMPILE_CHECKED = False
_GLOBAL_COMPILE_WORKS = False


def _check_compile_works() -> bool:
    """Runtime check to see if torch.compile actually works."""
    global _GLOBAL_COMPILE_CHECKED, _GLOBAL_COMPILE_WORKS

    if _GLOBAL_COMPILE_CHECKED:
        return _GLOBAL_COMPILE_WORKS

    if os.environ.get("BIOPL_DISABLE_COMPILE", "0") == "1":
        _GLOBAL_COMPILE_WORKS = False
        _GLOBAL_COMPILE_CHECKED = True
        return False

    try:

        def dummy_fn(x):
            return torch.tanh(x * 2.0)

        compiled = torch.compile(dummy_fn, mode="reduce-overhead")
        _ = compiled(torch.ones(128, 128))
        _GLOBAL_COMPILE_WORKS = True
    except Exception as e:
        warnings.warn(
            f"torch.compile check failed: {e}. Disabling compilation.",
            RuntimeWarning,
        )
        _GLOBAL_COMPILE_WORKS = False

    _GLOBAL_COMPILE_CHECKED = True
    return _GLOBAL_COMPILE_WORKS


def compile_model(
    model: torch.nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool | None = None,
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
        >>> model = compile_model(model, mode="reduce-overhead")
    """
    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile not available (requires PyTorch 2.0+). "
            "Using uncompiled model.",
            RuntimeWarning,
        )
        return model

    if not _check_compile_works():
        return model

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
            f"torch.compile failed: {e}. Using uncompiled model.",
            RuntimeWarning,
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

    if not torch.cuda.is_available():
        return settling_fn

    if not _check_compile_works():
        return settling_fn

    if not TRITON_AVAILABLE:
        return settling_fn

    try:
        return torch.compile(settling_fn, mode="reduce-overhead")
    except Exception:
        return settling_fn


__all__ = [
    "compile_model",
    "compile_settling_loop",
]
