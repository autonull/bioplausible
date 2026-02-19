"""
CUDA-accelerated operations for Dion SVD and Muon Newton-Schulz.

This module provides optimized CUDA kernels for:
1. Low-rank SVD (Dion updates)
2. Newton-Schulz iteration (Muon orthogonalization)
3. Spectral norm estimation via power iteration
4. Fused settling operations for EP
"""

import torch
from typing import Tuple, Optional, cast, List


def lowrank_svd_cuda(
    G: torch.Tensor,
    rank: int,
    q: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute low-rank SVD using CUDA-accelerated torch.svd_lowrank.

    Args:
        G: Input gradient matrix (M, N).
        rank: Target rank for decomposition.
        q: Number of power iterations for accuracy (default: 2).

    Returns:
        Tuple of (U, S, Vh) where:
            - U: Left singular vectors (M, rank)
            - S: Singular values (rank,)
            - Vh: Right singular vectors transposed (rank, N)
    """
    if G.dim() != 2:
        raise ValueError(f"Input must be 2D, got {G.dim()}D")

    if q is None:
        q = 2  # Default power iterations for accuracy

    # Ensure rank doesn't exceed matrix dimensions
    max_rank = min(G.shape)
    actual_rank = min(rank, max_rank)

    # Use PyTorch's optimized svd_lowrank (uses CUDA when available)
    # torch.svd_lowrank returns U, S, V where V is (N, rank), not Vh
    U, S, V = torch.svd_lowrank(G, q=actual_rank, niter=q)

    # Return Vh (transpose V)
    return U, S, V.T


def dion_update_cuda(
    G: torch.Tensor,
    rank: int,
    error_buffer: Optional[torch.Tensor] = None,
    error_beta: float = 0.9,
    q: Optional[int] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute Dion low-rank update with error feedback.

    The Dion update decomposes G ≈ U @ diag(S) @ Vh and returns
    the scale-invariant update U @ Vh.T, accumulating the residual
    in the error buffer.

    Args:
        G: Input gradient matrix (M, N).
        rank: Target rank for decomposition.
        error_buffer: Accumulated error from previous steps.
        error_beta: Decay factor for error buffer.
        q: Number of power iterations for SVD.

    Returns:
        Tuple of (update, new_error_buffer):
            - update: Low-rank update matrix (M, N)
            - new_error_buffer: Updated error buffer (or None if not provided)
    """
    # Add error feedback if available
    if error_buffer is not None:
        G_effective = G + error_beta * error_buffer
    else:
        G_effective = G

    # Compute low-rank SVD
    U, S, Vh = lowrank_svd_cuda(G_effective, rank, q=q)

    # Reconstruct low-rank update (scale-invariant)
    # Note: We ignore singular values S for scale-invariance
    update = U @ Vh

    # Compute residual for error feedback
    if error_buffer is not None:
        residual = G_effective - update
        new_error_buffer = error_beta * error_buffer + residual
    else:
        new_error_buffer = None

    return update, new_error_buffer


def newton_schulz_cuda(
    G: torch.Tensor,
    steps: int = 5,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    Newton-Schulz orthogonalization on CUDA.

    Iteration: X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)

    This converges quadratically when the initial matrix has spectral norm < sqrt(3).
    We normalize by Frobenius norm to ensure convergence.

    Note: This implementation matches the CPU version in SMEPOptimizer.newton_schulz()

    Args:
        G: Input gradient matrix (M, N).
        steps: Number of Newton-Schulz iterations.
        epsilon: Small value for numerical stability.

    Returns:
        Orthogonalized matrix with same shape as G.
    """
    if G.dim() != 2:
        raise ValueError(f"Input must be 2D, got {G.dim()}D")

    r, c = G.shape
    transposed = False

    # For rectangular matrices, ensure r >= c for stability
    if r < c:
        G = G.T
        r, c = c, r
        transposed = True

    # Pre-normalize to ensure convergence (Frobenius norm)
    # This matches the CPU implementation exactly
    X = G.clone()
    norm = X.norm().clamp(min=1e-4, max=1e4)
    X = X / norm

    # Newton-Schulz iteration: X = 0.5 * X * (3I - X^T X)
    identity = torch.eye(c, device=G.device, dtype=G.dtype)
    for _ in range(steps):
        A = X.T @ X
        X = 0.5 * X @ (3 * identity - A)

    # Restore original orientation if transposed
    if transposed:
        X = X.T

    return cast(torch.Tensor, X)


def spectral_norm_power_iteration_cuda(
    W: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    niter: int = 3,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute spectral norm via power iteration on CUDA.

    Power iteration:
        v = W^T u / ||W^T u||
        u = W v / ||W v||
        σ = u^T W v

    Args:
        W: Weight matrix (can be >2D, will be flattened to 2D).
        u: Left singular vector (cached from previous call).
        v: Right singular vector (cached from previous call).
        niter: Number of power iterations.
        epsilon: Small value for numerical stability.

    Returns:
        Tuple of (spectral_norm, updated_u, updated_v).
    """
    # Handle convolutional weights (4D -> 2D)
    if W.dim() > 2:
        W = W.view(W.shape[0], -1)

    h, w = W.shape

    # Initialize singular vectors if not provided
    if u is None:
        u = torch.randn(h, device=W.device, dtype=W.dtype)
        u = u / (u.norm() + epsilon)
    if v is None:
        v = torch.randn(w, device=W.device, dtype=W.dtype)
        v = v / (v.norm() + epsilon)

    # Power iteration
    for _ in range(niter):
        # v = W^T u
        v = torch.mv(W.T, u)
        v = v / (v.norm() + epsilon)

        # u = W v
        u = torch.mv(W, v)
        u = u / (u.norm() + epsilon)

    # Compute spectral norm
    sigma = torch.dot(u, torch.mv(W, v)).abs()

    return cast(torch.Tensor, sigma), u, v


def enforce_spectral_constraint_cuda(
    W: torch.Tensor,
    gamma: float = 0.95,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    niter: int = 3,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enforce spectral norm constraint via power iteration and scaling.

    If σ(W) > γ, scale W by γ/σ to ensure σ(W) ≤ γ.

    Args:
        W: Weight matrix to constrain.
        gamma: Maximum allowed spectral norm.
        u: Cached left singular vector.
        v: Cached right singular vector.
        niter: Number of power iterations.
        epsilon: Small value for numerical stability.

    Returns:
        Tuple of (constrained_W, updated_u, updated_v).
    """
    # Estimate spectral norm
    sigma, u, v = spectral_norm_power_iteration_cuda(
        W, u, v, niter=niter, epsilon=epsilon
    )

    # Scale if necessary
    if sigma > gamma:
        scale = gamma / sigma
        W_constrained = W * scale
    else:
        W_constrained = W

    return W_constrained, u, v


# Convenience function for batched operations
def batched_newton_schulz_cuda(
    G_batch: torch.Tensor,
    steps: int = 5,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    Batched Newton-Schulz orthogonalization.

    Processes a batch of matrices in parallel for better GPU utilization.

    Args:
        G_batch: Batch of gradient matrices (B, M, N).
        steps: Number of Newton-Schulz iterations.
        epsilon: Small value for numerical stability.

    Returns:
        Batch of orthogonalized matrices (B, M, N).
    """
    if G_batch.dim() != 3:
        raise ValueError(f"Input must be 3D batch, got {G_batch.dim()}D")

    b, r, c = G_batch.shape
    transposed = False

    # Ensure r >= c for stability
    if r < c:
        G_batch = G_batch.transpose(1, 2)
        b, r, c = b, c, r
        transposed = True

    # Normalize each matrix in batch (Frobenius norm)
    norms = G_batch.norm(dim=(1, 2), keepdim=True).clamp(min=epsilon, max=1e4)
    X = G_batch / norms

    # Batched Newton-Schulz iteration
    # X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
    for _ in range(steps):
        # Compute X^T X (B x c x c)
        XtX = torch.bmm(X.transpose(1, 2), X)

        # Compute 3I - XtX
        # Create identity for each batch
        identity = torch.eye(c, device=X.device, dtype=X.dtype).unsqueeze(0).expand(b, c, c)
        three_I_minus_XtX = 3 * identity - XtX

        # Update: X = 0.5 * X @ (3I - X^T X)
        X = 0.5 * torch.bmm(X, three_I_minus_XtX)

    # Restore orientation if transposed
    if transposed:
        X = X.transpose(1, 2)

    return cast(torch.Tensor, X)


def fused_settle_step(
    states: List[torch.Tensor],
    momentum_buffers: List[torch.Tensor],
    grads: List[torch.Tensor],
    momentum: float = 0.5,
    lr: float = 0.1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Fused settling step: apply momentum and update states in one operation.

    This reduces kernel launch overhead by fusing the momentum buffer update
    and state update into a single conceptual operation (though PyTorch still
    launches separate kernels, the data stays in cache).

    Formula:
        buf = momentum * buf + grad
        state = state - lr * buf

    Args:
        states: List of state tensors to update.
        momentum_buffers: List of momentum buffer tensors.
        grads: List of gradient tensors.
        momentum: Momentum coefficient (default 0.5).
        lr: Learning rate for settling.

    Returns:
        Tuple of (new_states, new_momentum_buffers).
    """
    new_states = []
    new_buffers = []

    for state, buf, grad in zip(states, momentum_buffers, grads):
        if grad is None:
            new_states.append(state)
            new_buffers.append(buf)
            continue

        # Fused update: buf = momentum * buf + grad; state = state - lr * buf
        # Using torch.addcmul for better fusion potential
        new_buf = torch.add(buf, grad, alpha=1.0)
        new_buf = torch.lerp(buf, new_buf, 1.0 - momentum)  # momentum * buf + grad
        
        new_state = torch.add(state, new_buf, alpha=-lr)

        new_states.append(new_state)
        new_buffers.append(new_buf)

    return new_states, new_buffers


def fused_settle_step_inplace(
    states: List[torch.Tensor],
    momentum_buffers: List[torch.Tensor],
    grads: List[torch.Tensor],
    momentum: float = 0.5,
    lr: float = 0.1,
) -> None:
    """
    In-place fused settling step for maximum efficiency.

    Modifies states and momentum buffers in-place to avoid allocations.

    Args:
        states: List of state tensors to update (modified in-place).
        momentum_buffers: List of momentum buffers (modified in-place).
        grads: List of gradient tensors.
        momentum: Momentum coefficient (default 0.5).
        lr: Learning rate for settling.
    """
    for state, buf, grad in zip(states, momentum_buffers, grads):
        if grad is None:
            continue

        # Update momentum buffer in-place: buf = momentum * buf + grad
        buf.mul_(momentum).add_(grad)

        # Update state in-place: state = state - lr * buf
        state.add_(buf, alpha=-lr)
