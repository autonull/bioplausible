"""
Base strategy protocol definitions.

This module defines the abstract interfaces for all strategy types.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Callable, Any
import torch
import torch.nn as nn


class GradientStrategy(Protocol):
    """
    Strategy for computing gradients.
    
    Implementations define how gradients are computed and accumulated
    into model parameters (e.g., backpropagation, equilibrium propagation).
    """
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        **kwargs: Any
    ) -> None:
        """
        Compute and accumulate gradients into model parameters.
        
        Args:
            model: The neural network module.
            x: Input tensor.
            target: Target tensor (may be None for unsupervised).
            **kwargs: Additional arguments (loss_fn, energy_fn, etc.).
        """
        ...


class UpdateStrategy(Protocol):
    """
    Strategy for transforming gradients into parameter updates.
    
    Implementations define how raw gradients are transformed before
    being applied (e.g., plain SGD, Newton-Schulz orthogonalization,
    low-rank SVD, Fisher whitening).
    """
    
    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        """
        Transform raw gradient into update direction.
        
        Args:
            param: The parameter being updated.
            gradient: The raw (or augmented) gradient tensor.
            state: Optimizer state dict for this parameter.
            group_config: Parameter group configuration.
        
        Returns:
            Transformed update tensor with same shape as gradient.
        """
        ...


class ConstraintStrategy(Protocol):
    """
    Strategy for enforcing parameter constraints.
    
    Implementations define constraints applied to parameters after
    updates (e.g., spectral norm bounds, Frobenius norm clipping).
    """
    
    def enforce(
        self,
        param: nn.Parameter,
        state: dict,
        group_config: dict
    ) -> None:
        """
        Enforce constraint on parameter in-place.
        
        Args:
            param: The parameter to constrain.
            state: Optimizer state dict for this parameter.
            group_config: Parameter group configuration.
        """
        ...


class FeedbackStrategy(Protocol):
    """
    Strategy for error/residual accumulation.
    
    Implementations define how update residuals are accumulated and
    fed back into future gradients (useful for low-rank approximations
    and continual learning).
    """
    
    def accumulate(
        self,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        """
        Accumulate residual and return augmented gradient.
        
        Args:
            gradient: The current gradient tensor.
            state: Optimizer state dict for this parameter.
            group_config: Parameter group configuration.
        
        Returns:
            Augmented gradient (gradient + accumulated error).
        """
        ...
    
    def update_buffer(
        self,
        residual: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> None:
        """
        Update error buffer with new residual.
        
        Args:
            residual: The update residual (gradient - applied_update).
            state: Optimizer state dict for this parameter.
            group_config: Parameter group configuration.
        """
        ...
