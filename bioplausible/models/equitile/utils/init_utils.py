from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def initialize_edge_weights(
    weight: Tensor,
    bias: Optional[Tensor] = None,
    init_type: str = "normal",
    gain: float = 1.0,
    nonlinearity: str = "relu",
    deep_init: bool = False,
    num_layers: Optional[int] = None,
) -> None:
    """Initialize edge weights and biases.

    Parameters
    ----------
    weight : Tensor
        Weight tensor to initialize.
    bias : Optional[Tensor]
        Bias tensor to initialize.
    init_type : str
        Initialization type ('normal', 'kaiming', 'xavier').
    gain : float
        Gain for initialization.
    nonlinearity : str
        Nonlinearity for kaiming initialization.
    deep_init : bool
        Whether to use deep initialization scaling.
    num_layers : Optional[int]
        Total number of layers, required for deep_init.
    """
    fan_in, fan_out = weight.shape[0], weight.shape[1]

    with torch.no_grad():
        if deep_init and num_layers is not None:
            # Deep network initialization
            depth_scale = math.sqrt(2.0 / (fan_in + fan_out))
            layer_factor = math.sqrt(2.0 / max(1, num_layers - 1))
            std = depth_scale * layer_factor * gain
            weight.normal_(0, std)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(
                weight, mode="fan_in", nonlinearity=nonlinearity
            )
            # Adjust gain manually if needed, kaiming handles it based on nonlinearity
            if gain != 1.0:
                 weight.data *= gain
        elif init_type == "xavier":
             nn.init.xavier_normal_(weight, gain=gain)
        else:
            # Default normal initialization based on fan_in
            std = math.sqrt(2.0 / fan_in) * gain
            weight.normal_(0, std)

        if bias is not None:
            nn.init.zeros_(bias)


def initialize_io_projections(
    w_in: nn.Linear,
    w_out: nn.Linear,
    deep_init: bool = False,
    num_layers: Optional[int] = None,
) -> None:
    """Initialize input and output projections.

    Parameters
    ----------
    w_in : nn.Linear
        Input projection layer.
    w_out : nn.Linear
        Output projection layer.
    deep_init : bool
        Whether to use deep initialization scaling.
    num_layers : Optional[int]
        Total number of layers, required for deep_init.
    """
    with torch.no_grad():
        nn.init.kaiming_normal_(
            w_in.weight, mode="fan_in", nonlinearity="relu"
        )
        if w_in.bias is not None:
            nn.init.zeros_(w_in.bias)

        if deep_init and num_layers is not None:
            # Scale output projection for deep networks
            output_scale = math.sqrt(2.0 / num_layers)
            nn.init.xavier_normal_(w_out.weight, gain=output_scale)
        else:
            nn.init.xavier_normal_(w_out.weight, gain=1.0)

        if w_out.bias is not None:
            nn.init.zeros_(w_out.bias)
