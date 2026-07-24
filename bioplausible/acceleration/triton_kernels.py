"""
Triton Kernels for EqProp Acceleration

Provides fused kernels for Equilibrium Propagation dynamics to maximize
GPU throughput by reducing memory bandwidth usage.
"""

import math
from typing import Optional

import torch

from bioplausible.acceleration.backends import HAS_CUPY
from bioplausible.acceleration.backends import HAS_TRITON


class TritonEqPropOps:
    """EqProp operations with optional Triton/CUDA acceleration.

    Provides step functions for equilibrium propagation with automatic
    fallback to PyTorch when Triton is not available.
    """

    _triton_kernel = None

    @classmethod
    def is_available(cls) -> bool:
        return HAS_TRITON

    @classmethod
    def _init_triton(cls):
        if cls._triton_kernel is None and HAS_TRITON:
            try:
                import triton
                import triton.language as tl

                @triton.jit
                def _step_kernel(
                    h_ptr,
                    pre_act_ptr,
                    bias_ptr,
                    out_ptr,
                    alpha,
                    n_elements,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    pid = tl.program_id(0)
                    block_start = pid * BLOCK_SIZE
                    offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements

                    h = tl.load(h_ptr + offsets, mask=mask)
                    pre_act = tl.load(pre_act_ptr + offsets, mask=mask)
                    bias = (
                        tl.load(bias_ptr + offsets, mask=mask)
                        if bias_ptr is not None
                        else 0.0
                    )

                    out = (1.0 - alpha) * h + alpha * tl.tanh(pre_act + bias)
                    tl.store(out_ptr + offsets, out, mask=mask)

                cls._triton_kernel = _step_kernel
            except ImportError:
                cls._triton_kernel = False

    @classmethod
    def step(
        cls,
        h: torch.Tensor,
        pre_act: torch.Tensor,
        alpha: float,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if HAS_TRITON and h.is_cuda and pre_act.is_cuda:
            cls._init_triton()
            if cls._triton_kernel:
                out = torch.empty_like(h)
                n_elements = h.numel()
                BLOCK_SIZE = 1024
                grid = (math.ceil(n_elements / BLOCK_SIZE),)
                bias_ptr = bias if bias is not None else None
                cls._triton_kernel[grid](
                    h,
                    pre_act,
                    bias_ptr,
                    out,
                    alpha,
                    n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                return out

        bias_val = bias if bias is not None else 0.0
        return (1.0 - alpha) * h + alpha * torch.tanh(pre_act + bias_val)

    @classmethod
    def step_linear(
        cls,
        h: torch.Tensor,
        h_target: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        return (1.0 - alpha) * h + alpha * h_target

    @classmethod
    def step_linear_cupy(cls, h, h_target, alpha):
        if not HAS_CUPY:
            raise ImportError("CuPy not available")
        return cls.step_linear(h, h_target, alpha)


__all__ = [
    "TritonEqPropOps",
    "HAS_TRITON",
    "HAS_CUPY",
]
