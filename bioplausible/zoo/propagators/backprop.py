"""
Standard autograd (backpropagation) wrapper.

Classes: Backprop
"""

import torch

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("backprop")
class Backprop(LearningRuleOptimizer):
    """Standard backpropagation via autograd."""

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        raise NotImplementedError
