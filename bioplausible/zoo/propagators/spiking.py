"""
Spiking / STDP propagators.

Classes: STDP
"""

import torch

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("stdp")
class STDP(LearningRuleOptimizer):
    """Spike-Timing-Dependent Plasticity."""

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        raise NotImplementedError
