"""
Constraint strategies for parameter enforcement.

Implements various constraint methods:
- No constraint (unconstrained optimization)
- Spectral norm constraint via power iteration
- Settling-time spectral penalty
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from .base import ConstraintStrategy

# Import CUDA kernels if available
try:
    from ...cuda.kernels import spectral_norm_power_iteration_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class NoConstraint:
    """
    No parameter constraints.
    
    Parameters are updated freely without any constraints.
    """
    
    def enforce(
        self,
        param: nn.Parameter,
        state: dict,
        group_config: dict
    ) -> None:
        pass


class SpectralConstraint:
    """
    Enforce spectral norm bound via power iteration.
    
    If σ(W) > γ, scales W by γ/σ to ensure σ(W) ≤ γ.
    
    This guarantees contractive dynamics and unique fixed points
    in equilibrium propagation.
    """
    
    EPSILON = 1e-6
    POWER_ITER = 3
    
    def __init__(
        self,
        gamma: float = 0.95,
        power_iter: int = 3,
        timing: str = "post_update",
    ):
        """
        Initialize spectral constraint.

        Args:
            gamma: Maximum allowed spectral norm (must be in (0, 1]).
            power_iter: Number of power iterations for estimation.
            timing: When to apply constraint ('post_update', 'during_settling', 'both').
        """
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if timing not in ("post_update", "during_settling", "both"):
            raise ValueError(f"Spectral timing must be 'post_update', 'during_settling', or 'both', got '{timing}'")

        self.gamma = gamma
        self.power_iter = power_iter
        self.timing = timing
    
    def enforce(
        self,
        param: nn.Parameter,
        state: dict,
        group_config: dict
    ) -> None:
        """
        Enforce spectral norm constraint on parameter.
        
        Only applies to 2D+ parameters (Linear, Conv weights).
        """
        if param.ndim < 2:
            return
        
        # Get cached singular vectors
        u = state.get("u_spec")
        v = state.get("v_spec")
        
        # Compute spectral norm via power iteration
        sigma, u, v = self._power_iteration(param.data, u, v)
        
        # Update cached vectors
        state["u_spec"] = u.detach()
        state["v_spec"] = v.detach()
        
        # Scale if necessary
        if sigma > self.gamma:
            param.data.mul_(self.gamma / sigma)
    
    def _power_iteration(
        self,
        W: torch.Tensor,
        u: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate spectral norm via power iteration.
        
        Returns:
            Tuple of (spectral_norm, updated_u, updated_v).
        """
        # Use CUDA if available
        if CUDA_AVAILABLE and W.is_cuda:
            return spectral_norm_power_iteration_cuda(
                W, u, v, niter=self.power_iter, epsilon=self.EPSILON
            )
        
        # Flatten for conv weights
        if W.ndim > 2:
            W = W.view(W.shape[0], -1)
        
        h, w = W.shape
        
        # Initialize singular vectors
        if u is None:
            u = torch.randn(h, device=W.device, dtype=W.dtype)
            u = u / (u.norm() + self.EPSILON)
        if v is None:
            v = torch.randn(w, device=W.device, dtype=W.dtype)
            v = v / (v.norm() + self.EPSILON)
        
        # Power iteration
        for _ in range(self.power_iter):
            v = W.T @ u
            v = v / (v.norm() + self.EPSILON)
            u = W @ v
            u = u / (u.norm() + self.EPSILON)
        
        sigma = (u @ W @ v).abs()
        return sigma, u, v
    
    def should_apply(self, timing: str) -> bool:
        """Check if constraint should be applied at given timing."""
        if self.timing == "both":
            return True
        return self.timing == timing


class SettlingSpectralPenalty:
    """
    Spectral penalty added during settling energy computation.
    
    Unlike SpectralConstraint which enforces a hard bound post-update,
    this adds a soft penalty to the energy function during settling:
    
        E_total = E_original + λ * Σ max(0, σ(W) - γ)²
    """
    
    def __init__(
        self,
        gamma: float = 0.95,
        lambda_penalty: float = 1.0,
    ):
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty
    
    def compute_penalty(
        self,
        model: nn.Module,
        optimizer_state: dict
    ) -> torch.Tensor:
        """
        Compute spectral penalty term for energy function.
        
        Args:
            model: Model with parameters to penalize.
            optimizer_state: Optimizer state dict for caching.
        
        Returns:
            Scalar penalty tensor to add to energy.
        """
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for param in model.parameters():
            if param.ndim < 2:
                continue
            
            state = optimizer_state.get(id(param), {})
            u = state.get("u_spec") if state else None
            v = state.get("v_spec") if state else None
            
            # Estimate spectral norm
            sigma, u, v = self._power_iteration(param.data, u, v)
            
            if state:
                state["u_spec"] = u.detach()
                state["v_spec"] = v.detach()
            
            if sigma > self.gamma:
                diff = sigma - self.gamma
                penalty = penalty + self.lambda_penalty * (diff ** 2)
        
        return penalty
    
    def _power_iteration(
        self,
        W: torch.Tensor,
        u: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        niter: int = 3,
        epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate spectral norm via power iteration."""
        if W.ndim > 2:
            W = W.view(W.shape[0], -1)
        
        h, w = W.shape
        
        if u is None:
            u = torch.randn(h, device=W.device, dtype=W.dtype)
            u = u / (u.norm() + epsilon)
        if v is None:
            v = torch.randn(w, device=W.device, dtype=W.dtype)
            v = v / (v.norm() + epsilon)
        
        for _ in range(niter):
            v = W.T @ u
            v = v / (v.norm() + epsilon)
            u = W @ v
            u = u / (u.norm() + epsilon)
        
        sigma = (u @ W @ v).abs()
        return sigma, u, v
