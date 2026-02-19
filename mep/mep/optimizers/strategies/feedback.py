"""
Feedback strategies for error/residual accumulation.

Implements various feedback methods:
- No feedback (standard optimization)
- Error feedback (accumulate residuals for future updates)
"""

import torch
import torch.nn as nn
from typing import Any, cast
from .base import FeedbackStrategy


class NoFeedback:
    """
    No error accumulation.
    
    Gradients are used directly without any residual accumulation.
    """
    
    def accumulate(
        self,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        return gradient
    
    def update_buffer(
        self,
        residual: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> None:
        pass


class ErrorFeedback:
    """
    Accumulate update residuals with decay.
    
    The residual (gradient - applied_update) is accumulated in a buffer
    and added to future gradients. This is particularly useful for:
    
    - Low-rank approximations (Dion) where information is lost
    - Continual learning where gradient history matters
    - Improving convergence with approximate updates
    
    Update rule:
        g_aug = g + β * error_buffer
        error_buffer = β * error_buffer + (g_aug - update)
    """
    
    def __init__(self, beta: float = 0.9):
        """
        Initialize error feedback.
        
        Args:
            beta: Decay factor for error buffer (must be in [0, 1)).
                  Higher values retain more history.
        """
        if not (0 <= beta < 1):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        
        self.beta = beta
    
    def accumulate(
        self,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        """
        Accumulate residual and return augmented gradient.
        
        Args:
            gradient: Current gradient tensor.
            state: Optimizer state dict for this parameter.
            group_config: Parameter group configuration.
        
        Returns:
            Augmented gradient (gradient + accumulated error).
        """
        # Initialize buffer if needed
        if "error_buffer" not in state:
            state["error_buffer"] = torch.zeros_like(gradient)
        
        buffer = cast(torch.Tensor, state["error_buffer"])
        return gradient + self.beta * buffer
    
    def update_buffer(
        self,
        residual: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> None:
        """
        Update error buffer with new residual.
        
        Args:
            residual: Update residual (gradient - applied_update).
            state: Optimizer state dict for this parameter.
            group_config: Parameter group configuration.
        """
        if "error_buffer" not in state:
            state["error_buffer"] = torch.zeros_like(residual)
        
        buffer = state["error_buffer"]
        buffer.mul_(self.beta).add_(residual)
        
        # Clip buffer to prevent explosion
        max_norm = group_config.get("max_grad_norm", 10.0) * 2
        buffer.clamp_(-max_norm, max_norm)
