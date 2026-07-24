"""
Standard autograd (backpropagation) wrapper.

Classes: Backprop
"""

from typing import Optional

import torch

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("backprop")
class Backprop(LearningRuleOptimizer):
    """Standard backpropagation via autograd."""

    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError
