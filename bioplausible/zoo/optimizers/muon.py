"""
Muon/Dion optimizers.
"""

from bioplausible.core.registry import register_optimizer


@register_optimizer("muon")
class MuonUpdate:
    """Muon orthogonalization update."""

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        pass


@register_optimizer("dion")
class DionUpdate:
    """Dion low-rank update."""

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        pass
