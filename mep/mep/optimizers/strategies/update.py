"""
Update transformation strategies.

Implements various methods for transforming gradients into updates:
- Plain (vanilla SGD)
- Muon (Newton-Schulz orthogonalization)
- Dion (low-rank SVD)
- Fisher-whitened Muon
"""

import torch
import torch.nn as nn
from typing import Optional, Any, cast
from .base import UpdateStrategy

# Import CUDA kernels if available
try:
    from ...cuda.kernels import newton_schulz_cuda, dion_update_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class PlainUpdate:
    """
    Vanilla SGD update (no transformation).
    
    The gradient is used directly as the update direction.
    """
    
    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        return gradient


class MuonUpdate:
    """
    Newton-Schulz orthogonalization (Muon optimizer).
    
    Applies iterative orthogonalization to the gradient:
        X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
    
    This produces an orthogonal update matrix, improving conditioning.
    """
    
    def __init__(self, ns_steps: int = 5):
        self.ns_steps = ns_steps
    
    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        orig_shape = None
        if gradient.ndim > 2:
            orig_shape = gradient.shape
            gradient = gradient.view(gradient.shape[0], -1)
        elif gradient.ndim < 2:
            return gradient

        update = self._newton_schulz(gradient, self.ns_steps)
        
        if orig_shape is not None:
            update = update.view(orig_shape)

        return update
    
    def _newton_schulz(
        self,
        G: torch.Tensor,
        steps: int,
        epsilon: float = 1e-4
    ) -> torch.Tensor:
        """Newton-Schulz orthogonalization."""
        if CUDA_AVAILABLE and G.is_cuda:
            return cast(torch.Tensor, newton_schulz_cuda(G, steps=steps, epsilon=epsilon))
        
        r, c = G.shape
        transposed = False
        
        if r < c:
            G = G.T
            r, c = c, r
            transposed = True
        
        # Pre-normalize (Frobenius norm)
        X = G.clone()
        norm = X.norm().clamp(min=1e-4, max=1e4)
        X = X / norm
        
        # Iteration: X = 0.5 * X * (3I - X^T X)
        identity = torch.eye(c, device=G.device, dtype=G.dtype)
        for _ in range(steps):
            A = X.T @ X
            X = 0.5 * X @ (3 * identity - A)
        
        if transposed:
            X = X.T
        
        return cast(torch.Tensor, X)


class DionUpdate:
    """
    Low-rank SVD update with error feedback.
    
    For large matrices (numel > threshold), uses low-rank SVD:
        G ≈ U @ S @ V^T
        update = U @ V^T  (scale-invariant)
    
    For smaller matrices, falls back to Muon orthogonalization.
    """
    
    def __init__(
        self,
        rank_frac: float = 0.2,
        threshold: int = 100000,
        muon_fallback: Optional[MuonUpdate] = None,
    ):
        self.rank_frac = rank_frac
        self.threshold = threshold
        self.muon_fallback = muon_fallback or MuonUpdate()
    
    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        # Use gradient numel if param is None (for testing)
        numel = param.numel() if param is not None else gradient.numel()
        
        if numel <= self.threshold:
            return self.muon_fallback.transform_gradient(
                param, gradient, state, group_config
            )
        
        # Low-rank SVD for large matrices
        if gradient.ndim != 2:
            orig_shape = gradient.shape
            gradient = gradient.view(gradient.shape[0], -1)
        else:
            orig_shape = None
        
        rank = max(1, int(min(gradient.shape) * self.rank_frac))
        max_rank = min(gradient.shape)
        rank = min(rank, max_rank)
        
        try:
            # Gradient clipping
            max_norm = group_config.get("max_grad_norm", 10.0)
            grad_norm = gradient.norm()
            if grad_norm > max_norm:
                gradient = gradient * (max_norm / (grad_norm + 1e-8))
            
            # Low-rank SVD
            if CUDA_AVAILABLE and gradient.is_cuda:
                error_buf = state.get("error_buffer")
                error_beta = group_config.get("error_beta", 0.9)
                use_feedback = group_config.get("use_error_feedback", True)
                
                if use_feedback and error_buf is not None:
                    update, new_buf = dion_update_cuda(
                        gradient, rank=rank,
                        error_buffer=error_buf,
                        error_beta=error_beta
                    )
                    state["error_buffer"] = new_buf
                else:
                    update, _ = dion_update_cuda(gradient, rank=rank)
            else:
                U, S, V = torch.svd_lowrank(gradient, q=rank)
                update = U @ V.T
                
                # Error feedback on CPU
                if group_config.get("use_error_feedback", True):
                    residual = gradient - update
                    error_beta = group_config.get("error_beta", 0.9)
                    if "error_buffer" not in state:
                        state["error_buffer"] = torch.zeros_like(residual)
                    state["error_buffer"].mul_(error_beta).add_(residual)
            
            if orig_shape is not None:
                update = update.view(orig_shape)
            
            return update
            
        except (RuntimeError, torch.linalg.LinAlgError):
            # Fallback to Muon
            return self.muon_fallback.transform_gradient(
                param, gradient, state, group_config
            )


class FisherUpdate:
    """
    Fisher-whitened gradient with Muon orthogonalization.
    
    Applies natural gradient preconditioning:
        whitened = g @ (F + λI)^-1
    
    Then orthogonalizes via Newton-Schulz.
    """
    
    def __init__(
        self,
        damping: float = 1e-3,
        ns_steps: int = 5,
        use_diagonal: bool = False,
        beta: float = 0.95,
    ):
        self.damping = damping
        self.ns_steps = ns_steps
        self.use_diagonal = use_diagonal
        self.beta = beta
    
    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict
    ) -> torch.Tensor:
        # Handle ND tensors by flattening
        orig_shape = None
        if gradient.ndim > 2:
            orig_shape = gradient.shape
            gradient = gradient.view(gradient.shape[0], -1)
        elif gradient.ndim < 2:
            return gradient

        # Check for new Fisher estimate on parameter
        if hasattr(param, "fisher"):
            fisher_estimate = getattr(param, "fisher")
            delattr(param, "fisher")  # Consume it

            if "fisher" not in state:
                state["fisher"] = fisher_estimate
            else:
                state["fisher"].mul_(self.beta).add_(fisher_estimate, alpha=1 - self.beta)
        
        fisher = state.get("fisher")
        
        if fisher is not None:
            if self.use_diagonal:
                # Diagonal whitening
                F = fisher + self.damping
                whitened = gradient / F.unsqueeze(0)
            else:
                # Full whitening: solve (F + λI) @ X = g^T
                F = fisher + self.damping * torch.eye(
                    fisher.shape[0], device=fisher.device, dtype=fisher.dtype
                )
                try:
                    whitened = torch.linalg.solve(F, gradient.T).T
                    if torch.isnan(whitened).any():
                        whitened = gradient
                except RuntimeError:
                    whitened = gradient
        else:
            whitened = gradient
        
        update = self._newton_schulz(whitened, self.ns_steps)

        if orig_shape is not None:
            update = update.view(orig_shape)

        return update
    
    def _newton_schulz(
        self,
        G: torch.Tensor,
        steps: int,
        epsilon: float = 1e-4
    ) -> torch.Tensor:
        """Newton-Schulz orthogonalization."""
        if CUDA_AVAILABLE and G.is_cuda:
            return cast(torch.Tensor, newton_schulz_cuda(G, steps=steps, epsilon=epsilon))
        
        r, c = G.shape
        transposed = False
        
        if r < c:
            G = G.T
            r, c = c, r
            transposed = True
        
        X = G.clone()
        norm = X.norm().clamp(min=1e-4, max=1e4)
        X = X / norm
        
        identity = torch.eye(c, device=G.device, dtype=G.dtype)
        for _ in range(steps):
            A = X.T @ X
            X = 0.5 * X @ (3 * identity - A)
        
        if transposed:
            X = X.T
        
        return cast(torch.Tensor, X)
