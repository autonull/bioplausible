"""
Predictive Coding propagators.

Classes: PCN
"""

from typing import Optional

import torch
from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("pcn")
class PCN(LearningRuleOptimizer):
    """Predictive Coding Network."""

    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError
