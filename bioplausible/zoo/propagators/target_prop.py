"""
Target Propagation family.

Classes: TargetProp, DifferenceTargetProp
"""

from typing import Optional

import torch
from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("target_prop")
class TargetProp(LearningRuleOptimizer):
    """Target Propagation: layer-wise target propagation."""

    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError


@register_propagator("difference_target_prop")
class DifferenceTargetProp(LearningRuleOptimizer):
    """Difference Target Propagation."""

    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError
