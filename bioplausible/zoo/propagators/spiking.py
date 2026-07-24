"""
Spiking / STDP propagators.

Classes: STDP
"""

from typing import Optional

import torch
from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("stdp")
class STDP(LearningRuleOptimizer):
    """Spike-Timing-Dependent Plasticity."""

    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError
