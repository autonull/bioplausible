"""
EquiTile Kernels
================

Shared mathematical operations for EquiTile components.
These kernels are used by:
- Core EquiTile (core.py)
- Async EquiTile (async_execution.py)
- Distributed EquiTile (distributed.py)

Consolidating these operations ensures consistency across different execution modes.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


def compute_tile_prediction(
    inputs: List[Tensor],
    bias: Optional[Tensor] = None,
    output_shape: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute prediction from inputs.

    Prediction = sum(inputs) + bias
    
    Parameters
    ----------
    inputs : list of Tensor
        Input tensors (e.g. from neighbors)
    bias : Tensor, optional
        Bias tensor
    output_shape : tuple, optional
        Expected output shape (batch_size, neurons)
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    Tensor
        Prediction tensor
    """
    if not inputs:
        if bias is not None:
            # Only bias
            pred = bias.unsqueeze(0)
            if output_shape is not None and pred.shape[0] != output_shape[0]:
                # Broadcast batch dimension if needed
                pred = pred.expand(output_shape)
            return pred

        # No inputs, no bias -> zeros
        if output_shape is not None and device is not None:
            return torch.zeros(output_shape, device=device)

        # Fallback to scalar 0 if no shape provided
        return torch.tensor(0.0)

    pred = sum(inputs)
    
    if bias is not None:
        pred = pred + bias.unsqueeze(0)

    return pred


def compute_activity_update(
    activity: Tensor,
    error: Tensor,
    fwd_feedback: List[Tensor],
    importance: float,
    step_size: float,
    lambda_error: float,
    clamp_min: float,
    clamp_max: float,
    clamp: bool
) -> Tensor:
    """Compute activity update for relaxation.
    
    delta = step_size * importance * (error + lambda * activity + feedback)
    new_activity = clamp(activity - delta)
    
    Parameters
    ----------
    activity : Tensor
        Current activity
    error : Tensor
        Current prediction error
    fwd_feedback : list of Tensor
        Feedback from forward neighbors (dst.error @ weight.T)
    importance : float
        Tile importance (sigmoid value)
    step_size : float
        Integration step size
    lambda_error : float
        Weight of prediction error term in energy
    clamp_min : float
        Minimum activity value
    clamp_max : float
        Maximum activity value
    clamp : bool
        Whether to clamp activity

    Returns
    -------
    Tensor
        New activity tensor
    """
    grad = error + lambda_error * activity
    
    for feedback in fwd_feedback:
        grad = grad + feedback

    delta = step_size * importance * grad
    new_activity = activity - delta
    
    if clamp:
        new_activity = torch.clamp(new_activity, clamp_min, clamp_max)

    return new_activity


def compute_hebbian_update(
    src_act: Tensor,
    dst_err: Tensor,
    importance: float,
    batch_size: int
) -> Tuple[Tensor, Tensor]:
    """Compute Hebbian weight and bias updates.
    
    weight_update = importance * (src.T @ dst_err) / batch
    bias_update = importance * mean(dst_err)
    
    Returns
    -------
    tuple
        (weight_update, bias_update) - ready to be subtracted from parameters
    """
    weight_update = importance * (src_act.T @ dst_err) / batch_size
    bias_update = importance * dst_err.mean(dim=0) / batch_size
    
    return weight_update, bias_update


def compute_contrastive_hebbian_update(
    src_free: Tensor,
    dst_free: Tensor,
    src_nudged: Tensor,
    dst_nudged: Tensor,
    learning_rate: float,
    beta: float,
    batch_size: int
) -> Tuple[Tensor, Tensor]:
    """Compute contrastive Hebbian update for Equilibrium Propagation.

    update ~ (free_stats - nudged_stats) / beta

    Returns
    -------
    tuple
        (weight_update, bias_update)
    """
    weight_update = (learning_rate / beta) * (
        src_free.T @ dst_free - src_nudged.T @ dst_nudged
    ) / batch_size
    
    bias_update = (learning_rate / beta) * (
        dst_free - dst_nudged
    ).mean(dim=0) / batch_size
    
    return weight_update, bias_update
