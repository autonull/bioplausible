"""
Base class for all Bioplausible optimizers.
"""

from typing import Callable, Optional

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
