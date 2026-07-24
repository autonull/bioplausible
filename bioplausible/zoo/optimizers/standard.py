"""
Standard PyTorch optimizers.
"""

from torch.optim import SGD as TorchSGD
from torch.optim import Adam as TorchAdam
from torch.optim import AdamW as TorchAdamW

from bioplausible.core.registry import register_optimizer


@register_optimizer("sgd")
class SGD(TorchSGD):
    """SGD optimizer wrapper."""

    pass


@register_optimizer("adam")
class Adam(TorchAdam):
    """Adam optimizer wrapper."""

    pass


@register_optimizer("adamw")
class AdamW(TorchAdamW):
    """AdamW optimizer wrapper."""

    pass
