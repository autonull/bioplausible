"""
Predictive Coding propagators.

Classes: PCN
"""

import torch

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("pcn")
class PCN(LearningRuleOptimizer):
    """Predictive Coding Network."""

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        raise NotImplementedError
