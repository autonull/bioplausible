"""
EWC (Elastic Weight Consolidation) optimizer.
"""

from bioplausible.core.registry import register_optimizer


@register_optimizer("ewc")
class EWC:
    """Elastic Weight Consolidation."""

    def __init__(self, params, lr=0.01, ewc_lambda=0.1):
        self.params = params
        self.lr = lr
        self.ewc_lambda = ewc_lambda

    def step(self):
        pass

    def zero_grad(self):
        pass
