from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def spectral_linear(
    in_features: int, out_features: int, bias: bool = True, use_sn: bool = True
) -> nn.Module:
    """Create a linear layer with optional spectral normalization."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    return spectral_norm(layer) if use_sn else layer


def spectral_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    use_sn: bool = True,
) -> nn.Module:
    """Create a Conv2d layer with optional spectral normalization."""
    layer = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=bias
    )
    return spectral_norm(layer) if use_sn else layer


def _get_layer_weight(layer: nn.Module) -> Optional[torch.Tensor]:
    """Extract weight tensor from a layer."""
    if hasattr(layer, "parametrizations") and hasattr(layer.parametrizations, "weight"):
        return layer.weight
    elif hasattr(layer, "weight"):
        return layer.weight
    return None


def _reshape_weight_for_power_iteration(weight: torch.Tensor) -> torch.Tensor:
    """Reshape weight tensor for power iteration (conv weights to 2D matrix)."""
    return weight.reshape(weight.shape[0], -1) if weight.dim() > 2 else weight


def estimate_lipschitz(layer: nn.Module, iterations: int = 3) -> float:
    """
    Estimate Lipschitz constant (spectral norm) of a layer using power iteration.
    Works for Linear and Conv2d layers.
    """
    weight = _get_layer_weight(layer)
    if weight is None:
        return 0.0

    device = weight.device
    W = _reshape_weight_for_power_iteration(weight)

    with torch.no_grad():
        u = F.normalize(torch.randn(W.shape[1], device=device), dim=0)

        for _ in range(iterations):
            v = F.normalize(torch.mv(W, u), dim=0)
            u = F.normalize(torch.mv(W.t(), v), dim=0)

        sigma = torch.dot(u, torch.mv(W.t(), v))

    return sigma.item()


def _has_spectral_norm(layer: nn.Module) -> bool:
    """Check if a module has spectral normalization."""
    return hasattr(layer, "parametrizations") and hasattr(
        layer.parametrizations, "weight"
    )
