"""
MEP Benchmark Baselines

Provides factory function to get optimizer instances by name.
Includes both standard PyTorch optimizers (baselines) and EP-based optimizers.
"""

from typing import Tuple, Any, Dict
import torch.nn as nn
import torch.optim as optim
from mep import smep, sdmep, local_ep, natural_ep, muon_backprop


def get_optimizer(
    name: str,
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    **kwargs: Any
) -> Tuple[Any, bool]:
    """
    Get optimizer by name.

    Args:
        name: Optimizer name (case-insensitive). Options:
            - 'sgd': SGD with momentum (baseline)
            - 'adam': Adam optimizer (baseline)
            - 'adamw': AdamW optimizer with decoupled weight decay (baseline)
            - 'muon': Standalone Muon optimizer (backprop mode)
            - 'eqprop': Vanilla Equilibrium Propagation (no spectral/Muon)
            - 'smep': Spectral Muon EP (Muon + EP gradients)
            - 'sdmep': Full SDMEP (Dion + Muon + EP)
            - 'local_ep': Local EP with layer-local updates
            - 'natural_ep': Natural EP with Fisher whitening
        model: PyTorch model to optimize
        lr: Learning rate
        momentum: Momentum for SGD-based optimizers
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Tuple of (optimizer_instance, is_ep_optimizer) where:
            - optimizer_instance: The optimizer object
            - is_ep_optimizer: True if EP gradients are required
    """
    name = name.lower()
    params = model.parameters()

    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay), False

    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay), False

    if name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay), False

    if name == 'muon':
        # Standalone Muon: backprop with Muon orthogonalization
        return muon_backprop(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_steps=kwargs.get('ns_steps', 5),
            gamma=kwargs.get('gamma', 0.95)
        ), False

    if name == 'eqprop':
        # Vanilla EP: SMEP with no Newton-Schulz orthogonalization
        return smep(
            params, model=model, lr=lr, momentum=momentum, weight_decay=weight_decay,
            mode='ep', ns_steps=0,
            beta=kwargs.get('beta', 0.5),
            settle_steps=kwargs.get('settle_steps', 10),
            settle_lr=kwargs.get('settle_lr', 0.05),
            loss_type=kwargs.get('loss_type', 'mse'),
            use_error_feedback=False
        ), True

    if name == 'smep':
        # Spectral Muon EP
        return smep(
            params, model=model, lr=lr, momentum=momentum, weight_decay=weight_decay,
            mode='ep',
            beta=kwargs.get('beta', 0.5),
            settle_steps=kwargs.get('settle_steps', 10),
            settle_lr=kwargs.get('settle_lr', 0.05),
            gamma=kwargs.get('gamma', 0.95),
            ns_steps=kwargs.get('ns_steps', 5),
            error_beta=kwargs.get('error_beta', 0.9),
            use_error_feedback=kwargs.get('use_error_feedback', False),  # Disabled for stability
            loss_type=kwargs.get('loss_type', 'mse')
        ), True

    if name == 'sdmep':
        # Full SDMEP with Dion for large matrices
        return sdmep(
            params, model=model, lr=lr, momentum=momentum, weight_decay=weight_decay,
            mode='ep',
            beta=kwargs.get('beta', 0.5),
            settle_steps=kwargs.get('settle_steps', 10),
            settle_lr=kwargs.get('settle_lr', 0.05),
            gamma=kwargs.get('gamma', 0.95),
            rank_frac=kwargs.get('rank_frac', 0.2),
            dion_thresh=kwargs.get('dion_thresh', 100000),
            error_beta=kwargs.get('error_beta', 0.9),
            use_error_feedback=False,  # Disabled for classification stability
            loss_type=kwargs.get('loss_type', 'mse')
        ), True

    if name == 'local_ep':
        # Local EP with layer-local updates
        return local_ep(
            params, model=model, lr=lr, momentum=momentum, weight_decay=weight_decay,
            beta=kwargs.get('beta', 0.1),
            settle_steps=kwargs.get('settle_steps', 10),
            settle_lr=kwargs.get('settle_lr', 0.05),
            gamma=kwargs.get('gamma', 0.95)
        ), True

    if name == 'natural_ep':
        # Natural EP with Fisher whitening
        return natural_ep(
            params, model=model, lr=lr, momentum=momentum, weight_decay=weight_decay,
            beta=kwargs.get('beta', 0.5),
            settle_steps=kwargs.get('settle_steps', 10),
            settle_lr=kwargs.get('settle_lr', 0.05),
            gamma=kwargs.get('gamma', 0.95),
            fisher_approx=kwargs.get('fisher_approx', 'empirical'),
            fisher_damping=kwargs.get('fisher_damping', 1e-3)
        ), True

    raise ValueError(f"Unknown optimizer: {name}. Available: sgd, adam, adamw, muon, eqprop, smep, sdmep, local_ep, natural_ep")
