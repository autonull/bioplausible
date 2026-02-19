"""
Preset optimizer configurations.

Factory functions for common optimizer combinations.
"""

import torch.nn as nn
from typing import Iterable, Optional, Any

from mep.optimizers import (
    CompositeOptimizer,
    BackpropGradient,
    EPGradient,
    LocalEPGradient,
    NaturalGradient,
    PlainUpdate,
    MuonUpdate,
    DionUpdate,
    FisherUpdate,
    NoConstraint,
    SpectralConstraint,
    NoFeedback,
    ErrorFeedback,
    GradientStrategy
)


def smep(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    mode: str = "backprop",
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    ns_steps: int = 5,
    beta: float = 0.5,  # Increased from 0.3 for better convergence
    settle_steps: int = 30,  # Increased from 15 for better settling
    settle_lr: float = 0.15,  # Increased from 0.1 for faster settling
    gamma: float = 0.95,
    spectral_timing: str = "post_update",
    error_beta: float = 0.9,
    use_error_feedback: bool = False,  # Disabled by default - only for Dion/CL
    loss_type: str = "mse",  # MSE for stable EP energy
    softmax_temperature: float = 1.0,
    **kwargs: Any
) -> CompositeOptimizer:
    """
    SMEP: Spectral Muon Equilibrium Propagation.

    Combines:
    - EP or backprop gradients
    - Muon (Newton-Schulz) orthogonalization
    - Spectral norm constraints
    - Optional error feedback (for Dion updates or continual learning)

    Recommended defaults for classification (tuned for MNIST):
    - lr=0.01, beta=0.5, settle_steps=30, settle_lr=0.15
    - use_error_feedback=False for standard training
    - loss_type='mse' for stable EP energy computation

    Note: EP achieves ~95% on MNIST with these settings, matching Adam.
    
    Speed note: Default settle_steps=30 gives ~10-15x slower training vs backprop.
    For faster training, use settle_steps=10-15 (4-6x slower) or O1MemoryEPv2 (3-5x slower).

    Args:
        params: Parameters to optimize.
        model: Model instance.
        mode: 'backprop' or 'ep'.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay.
        ns_steps: Newton-Schulz iterations.
        beta: EP nudging strength (0.3-0.7 typical, 0.5 recommended).
        settle_steps: EP settling iterations (20-50 typical, 30 recommended).
        settle_lr: Settling learning rate (0.1-0.2 typical, 0.15 recommended).
        gamma: Spectral norm bound.
        spectral_timing: When to apply spectral constraint.
        error_beta: Error feedback decay.
        use_error_feedback: Enable error feedback (for Dion/CL only).
        loss_type: 'mse' or 'cross_entropy'.
        softmax_temperature: Temperature for softmax in classification.

    Returns:
        Configured CompositeOptimizer.
    """
    gradient: GradientStrategy

    # Gradient strategy
    if mode == "ep":
        gradient = EPGradient(
            beta=beta,
            settle_steps=settle_steps,
            settle_lr=settle_lr,
            loss_type=loss_type,
            softmax_temperature=softmax_temperature,
        )
    else:
        gradient = BackpropGradient()

    # Update strategy
    update = MuonUpdate(ns_steps=ns_steps)

    # Constraint strategy
    constraint = SpectralConstraint(
        gamma=gamma,
        timing=spectral_timing,
    )

    # Feedback strategy
    feedback = ErrorFeedback(beta=error_beta) if use_error_feedback else NoFeedback()

    return CompositeOptimizer(
        params,
        gradient=gradient,
        update=update,
        constraint=constraint,
        feedback=feedback,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        model=model,
        **kwargs
    )


def sdmep(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    mode: str = "ep",
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    ns_steps: int = 5,
    beta: float = 0.3,
    settle_steps: int = 15,
    settle_lr: float = 0.1,
    gamma: float = 0.95,
    rank_frac: float = 0.2,
    dion_thresh: int = 100000,
    error_beta: float = 0.9,
    use_error_feedback: bool = True,  # Enabled for Dion (recovers lost info)
    loss_type: str = "cross_entropy",
    softmax_temperature: float = 1.0,
    **kwargs: Any
) -> CompositeOptimizer:
    """
    SDMEP: Spectral Dion-Muon Equilibrium Propagation.

    Like SMEP but uses low-rank SVD (Dion) for large matrices.
    Error feedback is enabled by default to recover information lost
    in low-rank approximation.

    Recommended defaults for classification:
    - lr=0.01, beta=0.3, settle_steps=15, settle_lr=0.1
    - For large models: dion_thresh=200000, rank_frac=0.15

    Args:
        params: Parameters to optimize.
        model: Model instance.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay.
        ns_steps: Newton-Schulz iterations.
        beta: EP nudging strength (0.1-0.5 typical).
        settle_steps: EP settling iterations (10-30 typical).
        settle_lr: Settling learning rate (0.05-0.2 typical).
        gamma: Spectral norm bound.
        rank_frac: Fraction of singular values to retain.
        dion_thresh: Parameter threshold for Dion vs Muon.
        error_beta: Error feedback decay.
        use_error_feedback: Enable error feedback for Dion (default: True).
        loss_type: 'mse' or 'cross_entropy'.

    Returns:
        Configured CompositeOptimizer.
    """
    gradient = EPGradient(
        beta=beta,
        settle_steps=settle_steps,
        settle_lr=settle_lr,
        loss_type=loss_type,
    )

    update = DionUpdate(
        rank_frac=rank_frac,
        threshold=dion_thresh,
        muon_fallback=MuonUpdate(ns_steps=ns_steps),
    )

    constraint = SpectralConstraint(gamma=gamma)

    feedback = ErrorFeedback(beta=error_beta) if use_error_feedback else NoFeedback()

    return CompositeOptimizer(
        params,
        gradient=gradient,
        update=update,
        constraint=constraint,
        feedback=feedback,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        model=model,
        **kwargs
    )


def local_ep(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float = 0.02,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    ns_steps: int = 5,
    beta: float = 0.1,
    settle_steps: int = 20,
    settle_lr: float = 0.05,
    gamma: float = 0.95,
    loss_type: str = "mse",
    **kwargs: Any
) -> CompositeOptimizer:
    """
    LocalEPMuon: Layer-local EP with Muon orthogonalization.
    
    Biologically plausible: each layer updates using only local information.
    
    Args:
        params: Parameters to optimize.
        model: Model instance.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay.
        ns_steps: Newton-Schulz iterations.
        beta: EP nudging strength.
        settle_steps: EP settling iterations.
        settle_lr: Settling learning rate.
        gamma: Spectral norm bound.
        loss_type: 'mse' or 'cross_entropy'.
    
    Returns:
        Configured CompositeOptimizer.
    """
    gradient = LocalEPGradient(
        beta=beta,
        settle_steps=settle_steps,
        settle_lr=settle_lr,
        loss_type=loss_type,
    )
    
    update = MuonUpdate(ns_steps=ns_steps)
    constraint = SpectralConstraint(gamma=gamma)
    feedback = NoFeedback()  # Local EP doesn't use error feedback
    
    return CompositeOptimizer(
        params,
        gradient=gradient,
        update=update,
        constraint=constraint,
        feedback=feedback,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        model=model,
        **kwargs
    )


def natural_ep(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float = 0.02,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    ns_steps: int = 5,
    beta: float = 0.5,
    settle_steps: int = 20,
    settle_lr: float = 0.05,
    gamma: float = 0.95,
    fisher_approx: str = "empirical",
    fisher_damping: float = 1e-3,
    use_diagonal_fisher: bool = False,
    loss_type: str = "mse",
    **kwargs: Any
) -> CompositeOptimizer:
    """
    NaturalEPMuon: Natural gradient EP with Fisher whitening.
    
    Uses Fisher Information Matrix for geometry-aware updates.
    
    Args:
        params: Parameters to optimize.
        model: Model instance.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay.
        ns_steps: Newton-Schulz iterations.
        beta: EP nudging strength.
        settle_steps: EP settling iterations.
        settle_lr: Settling learning rate.
        gamma: Spectral norm bound.
        fisher_approx: Fisher approximation method.
        fisher_damping: Damping for Fisher matrix inversion.
        use_diagonal_fisher: Use diagonal Fisher approximation.
        loss_type: 'mse' or 'cross_entropy'.
    
    Returns:
        Configured CompositeOptimizer.
    """
    base_gradient = EPGradient(
        beta=beta,
        settle_steps=settle_steps,
        settle_lr=settle_lr,
        loss_type=loss_type,
    )
    
    gradient = NaturalGradient(
        base_strategy=base_gradient,
        fisher_approx=fisher_approx,
        use_diagonal=use_diagonal_fisher,
    )
    
    update = FisherUpdate(
        damping=fisher_damping,
        ns_steps=ns_steps,
        use_diagonal=use_diagonal_fisher,
    )
    
    constraint = SpectralConstraint(gamma=gamma)
    feedback = NoFeedback()
    
    return CompositeOptimizer(
        params,
        gradient=gradient,
        update=update,
        constraint=constraint,
        feedback=feedback,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        model=model,
        **kwargs
    )


def muon_backprop(
    params: Iterable[nn.Parameter],
    lr: float = 0.02,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    ns_steps: int = 5,
    gamma: float = 0.95,
    use_spectral: bool = True,
    **kwargs: Any
) -> CompositeOptimizer:
    """
    Muon optimizer with standard backpropagation.

    A drop-in replacement for SGD/Adam with Muon orthogonalization.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay.
        ns_steps: Newton-Schulz iterations.
        gamma: Spectral norm bound.
        use_spectral: Enable spectral constraints.

    Returns:
        Configured CompositeOptimizer.
    """
    gradient = BackpropGradient()
    update = MuonUpdate(ns_steps=ns_steps)
    constraint = SpectralConstraint(gamma=gamma) if use_spectral else NoConstraint()
    feedback = NoFeedback()

    return CompositeOptimizer(
        params,
        gradient=gradient,
        update=update,
        constraint=constraint,
        feedback=feedback,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs
    )


def smep_fast(
    params: Iterable[nn.Parameter],
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    ns_steps: int = 5,
    beta: float = 0.5,
    settle_steps: int = 10,  # Reduced from 30 for speed
    settle_lr: float = 0.2,  # Higher LR for faster convergence
    gamma: float = 0.95,
    spectral_timing: str = "post_update",
    error_beta: float = 0.9,
    use_error_feedback: bool = False,
    loss_type: str = "mse",
    softmax_temperature: float = 1.0,
    **kwargs: Any
) -> CompositeOptimizer:
    """
    SMEP-Fast: Optimized SMEP for faster training.

    Uses fewer settling steps (10 vs 30) and higher settling LR for
    faster convergence. Achieves 4-6x speedup vs default settings
    with minimal accuracy impact.

    Speed comparison:
    - Default SMEP (30 steps): ~10-15x slower than backprop
    - SMEP-Fast (10 steps): ~4-6x slower than backprop
    - O1MemoryEPv2 (analytic): ~3-5x slower than backprop

    Args:
        params: Parameters to optimize.
        model: Model instance.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay.
        ns_steps: Newton-Schulz iterations.
        beta: EP nudging strength.
        settle_steps: EP settling iterations (10 for speed).
        settle_lr: Settling learning rate (0.2 for faster convergence).
        gamma: Spectral norm bound.
        spectral_timing: When to apply spectral constraint.
        error_beta: Error feedback decay.
        use_error_feedback: Enable error feedback.
        loss_type: 'mse' or 'cross_entropy'.
        softmax_temperature: Temperature for softmax.

    Returns:
        Configured CompositeOptimizer.
    """
    gradient = EPGradient(
        beta=beta,
        settle_steps=settle_steps,
        settle_lr=settle_lr,
        loss_type=loss_type,
        softmax_temperature=softmax_temperature,
    )

    update = MuonUpdate(ns_steps=ns_steps)
    constraint = SpectralConstraint(gamma=gamma, timing=spectral_timing)
    feedback = ErrorFeedback(beta=error_beta) if use_error_feedback else NoFeedback()

    return CompositeOptimizer(
        params,
        gradient=gradient,
        update=update,
        constraint=constraint,
        feedback=feedback,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        model=model,
        **kwargs
    )
