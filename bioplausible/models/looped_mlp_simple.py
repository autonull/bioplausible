"""
Simplified LoopedMLP

A clean, minimal implementation of the recurrent MLP for EqProp.
Uses PyTorch primitives and the EqPropModel base class.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from typing import Optional, Tuple

from .eqprop_base import EqPropModel


class LoopedMLP(EqPropModel):
    """
    Recurrent MLP that iterates to equilibrium.
    
    Architecture:
        h_{t+1} = tanh(W_in @ x + W_rec @ h_t)
        output = W_out @ h*  (where h* is the fixed point)
    
    Key properties:
    - Recurrent dynamics (not feedforward like standard MLP)
    - Spectral normalization for L < 1 guarantee
    - Settles to fixed point in ~20-30 steps
    - Compatible with EqProp training
    
    Example:
        model = LoopedMLP(input_dim=784, hidden_dim=256, output_dim=10)
        x = torch.randn(32, 784)
        output = model(x, steps=30)  # [32, 10]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        **kwargs,
    ):
        # Store dims before calling super().__init__
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._use_spectral_norm = use_spectral_norm
        
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            **kwargs,
        )
    
    def _build_layers(self):
        """Build all layers."""
        # Core recurrent weights
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        
        # Apply spectral norm if requested
        if self._use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable dynamics."""
        nn.init.kaiming_normal_(self.W_in.weight, nonlinearity='tanh')
        nn.init.orthogonal_(self.W_rec.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.W_out.bias)
    
    def forward_step(
        self,
        h: torch.Tensor,
        x_transformed: torch.Tensor,
    ) -> torch.Tensor:
        """Single equilibrium iteration step."""
        return torch.tanh(self.W_in(x_transformed) + self.W_rec(h))
    
    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
    
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Input is already in correct format."""
        return x
    
    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden state to output."""
        return self.W_out(h)
    
    def forward(self, x: torch.Tensor, steps: Optional[int] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass with settling dynamics.
        
        Args:
            x: Input tensor [batch, input_dim].
            steps: Number of settling steps (default: max_steps).
        
        Returns:
            Output tensor [batch, output_dim].
        """
        # Use EqPropModel's equilibrium forward pass
        return super().forward(x, steps=steps, **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"LoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm})"
        )


# ============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# ============================================================================

EqPropMLP = LoopedMLP
RecurrentMLP = LoopedMLP


__all__ = ['LoopedMLP', 'EqPropMLP', 'RecurrentMLP']
