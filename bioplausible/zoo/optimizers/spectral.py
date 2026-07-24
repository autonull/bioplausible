"""
Spectral constraint optimizers.
"""

from bioplausible.core.registry import register_optimizer


@register_optimizer("spectral")
class SpectralConstraint:
    """Spectral constraint on weights."""

    def __init__(self, params, lr=0.01, max_norm=1.0):
        self.params = params
        self.lr = lr
        self.max_norm = max_norm

    def step(self):
        pass

    def zero_grad(self):
        pass
