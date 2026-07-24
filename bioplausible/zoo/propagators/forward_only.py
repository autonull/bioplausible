"""
Forward-only propagators.

Classes: FF, PEPITA
"""

import torch

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("ff")
class FF(LearningRuleOptimizer):
    """Forward-Forward learning rule."""

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        raise NotImplementedError


@register_propagator("pepita")
class PEPITA(LearningRuleOptimizer):
    """PEPITA: forward-only learning with random feedback."""

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        raise NotImplementedError
