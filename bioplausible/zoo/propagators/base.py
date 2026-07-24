"""
Base classes for all Bioplausible propagators (learning rules).

BioOptimizer: extends torch.optim.Optimizer for biologically plausible learning.
LearningRuleOptimizer: shared base for learning-rule-based propagators.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


class BioOptimizer(Optimizer):
    """
    Base class for all Bioplausible optimizers.

    Extends PyTorch's Optimizer to support biologically plausible learning.
    """

    def __init__(self, params, model: Optional[nn.Module] = None, **defaults):
        params = list(params)
        super().__init__(params, defaults)
        self.model = model
        self.params = params

    def step(self, closure: Optional[Callable] = None, **kwargs):
        """Perform optimization step."""
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients."""
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


class LearningRuleOptimizer(BioOptimizer):
    """
    Base class for learning rule optimizers.

    Learning rules define how model parameters are updated based on
    inputs, targets, and model states.
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
    ):
        super().__init__(
            params,
            model=model,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.buffers = [torch.zeros_like(p) for p in self.params]

    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def _apply_update(
        self, grad: torch.Tensor, param: nn.Parameter, buffer: torch.Tensor
    ) -> None:
        momentum = getattr(self, "momentum", self.defaults.get("momentum", 0.9))
        weight_decay = getattr(
            self, "weight_decay", self.defaults.get("weight_decay", 0.0005)
        )
        lr = getattr(self, "lr", self.defaults.get("lr", 0.01))

        buffer.mul_(momentum).add_(grad)

        if weight_decay > 0:
            param.data.mul_(1 - weight_decay * lr)

        param.data.add_(buffer, alpha=-lr)
