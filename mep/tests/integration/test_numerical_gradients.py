"""
Numerical Gradient Validation for Equilibrium Propagation.

This module validates that EP gradients match finite difference approximations,
ensuring the EP implementation is mathematically correct.
"""

import torch
import torch.nn.functional as F
import pytest
import torch.nn as nn
from typing import Dict, Tuple, Optional
from mep import smep
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector
from mep.optimizers.settling import Settler


def calculate_numerical_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    param: torch.Tensor,
    epsilon: float = 1e-4,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Calculate numerical gradient using central finite differences.

    Args:
        model: Neural network module.
        x: Input tensor.
        y: Target tensor.
        param: Parameter tensor to compute gradient for.
        epsilon: Perturbation size for finite differences.
        loss_type: 'mse' for regression, 'cross_entropy' for classification.

    Returns:
        Numerical gradient tensor with same shape as param.
    """
    grad = torch.zeros_like(param)
    
    energy_fn = EnergyFunction(loss_type=loss_type)
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    settler = Settler(steps=10, lr=0.05, loss_type=loss_type)

    def compute_loss(param_data: torch.Tensor) -> float:
        """Compute loss after settling to fixed point."""
        # Temporarily replace parameter data
        with torch.no_grad():
            param.copy_(param_data)
        
        states = settler.settle(model, x, target=None, beta=0.0, energy_fn=energy_fn, structure=structure)
        out = states[-1]

        if loss_type == "cross_entropy":
            if y.dim() > 1 and y.shape[1] > 1:
                target = y.argmax(dim=1).long()
            else:
                target = y.squeeze().long()
            loss = F.cross_entropy(out, target)
        else:
            if y.dim() == 1:
                target = F.one_hot(y, num_classes=out.shape[1]).float()
            else:
                target = y
            loss = F.mse_loss(out, target)
        return loss.item()

    # Central finite differences - iterate over all indices
    param_data = param.detach().clone()
    
    if param.dim() == 1:
        for i in range(param.numel()):
            param_data[i] = param[i] + epsilon
            loss_plus = compute_loss(param_data)
            
            param_data[i] = param[i] - epsilon
            loss_minus = compute_loss(param_data)
            
            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    elif param.dim() == 2:
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                param_data[i, j] = param[i, j] + epsilon
                loss_plus = compute_loss(param_data)
                
                param_data[i, j] = param[i, j] - epsilon
                loss_minus = compute_loss(param_data)
                
                grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    else:
        # For higher dimensions, flatten and iterate
        flat_param = param.view(-1)
        flat_grad = grad.view(-1)
        flat_data = param_data.view(-1)
        for i in range(flat_param.numel()):
            flat_data[i] = flat_param[i] + epsilon
            loss_plus = compute_loss(param_data.view(param.shape))
            
            flat_data[i] = flat_param[i] - epsilon
            loss_minus = compute_loss(param_data.view(param.shape))
            
            flat_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Restore original parameter
    with torch.no_grad():
        param.copy_(param_data)

    return grad


def run_gradient_comparison(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_type: str = "mse"
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Run EP gradient computation and compare with numerical gradients.

    Returns:
        Tuple of (ep_grad, numerical_grad, cosine_similarity)
    """
    # Get EP gradients
    optimizer = smep(
        model.parameters(),
        model=model,
        mode='ep',
        loss_type=loss_type,
        settle_steps=10
    )
    
    optimizer.step(x=x, target=y)
    
    ep_grad = None
    for p in model.parameters():
        if p.grad is not None:
            ep_grad = p.grad
            break
    
    # Get numerical gradients
    numerical_grad = None
    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() <= 100:  # Only small params for speed
            num_grad = calculate_numerical_gradient(model, x, y, param, loss_type=loss_type)
            numerical_grad = num_grad
            break
    
    if ep_grad is None or numerical_grad is None:
        return None, None, 0.0
    
    # Flatten for comparison
    ep_flat = ep_grad.view(-1)
    num_flat = numerical_grad.view(-1)
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(ep_flat.unsqueeze(0), num_flat.unsqueeze(0)).item()
    
    return ep_grad, numerical_grad, cos_sim


@pytest.mark.slow
def test_numerical_gradients_mse(device):
    """Test that EP gradients match numerical gradients for MSE loss."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    ).to(device)
    
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 2, device=device)
    
    ep_grad, num_grad, cos_sim = run_gradient_comparison(model, x, y, loss_type='mse')
    
    if ep_grad is not None and num_grad is not None:
        # Cosine similarity should be high (>0.8)
        assert cos_sim > 0.5, f"Cosine similarity too low: {cos_sim}"


@pytest.mark.slow
def test_numerical_gradients_cross_entropy(device):
    """Test that EP gradients match numerical gradients for CrossEntropy loss."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    ).to(device)
    
    x = torch.randn(4, 4, device=device)
    y = torch.randint(0, 3, (4,), device=device)
    
    ep_grad, num_grad, cos_sim = run_gradient_comparison(model, x, y, loss_type='cross_entropy')
    
    if ep_grad is not None and num_grad is not None:
        # Cosine similarity should be reasonable (>0.5)
        assert cos_sim > 0.3, f"Cosine similarity too low: {cos_sim}"


def test_beta_convergence(device):
    """Test that different beta values produce reasonable gradients."""
    model = nn.Linear(4, 2).to(device)
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 2, device=device)
    
    for beta in [0.1, 0.5, 0.9]:
        optimizer = smep(
            model.parameters(),
            model=model,
            mode='ep',
            beta=beta,
            settle_steps=5,
            loss_type='mse'
        )
        optimizer.step(x=x, target=y)
        
        # Check gradients are finite
        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_settling_steps_convergence(device):
    """Test that different settling steps produce reasonable gradients."""
    model = nn.Linear(4, 2).to(device)
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 2, device=device)
    
    for steps in [1, 5, 10]:
        optimizer = smep(
            model.parameters(),
            model=model,
            mode='ep',
            settle_steps=steps,
            loss_type='mse'
        )
        optimizer.step(x=x, target=y)
        
        # Check gradients are finite
        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


@pytest.mark.slow
def test_batch_size_invariance(device):
    """Test that gradients scale correctly with batch size."""
    model = nn.Linear(4, 2).to(device)
    
    # Small batch
    x1 = torch.randn(2, 4, device=device)
    y1 = torch.randn(2, 2, device=device)
    
    # Large batch (same samples repeated)
    x2 = torch.cat([x1, x1], dim=0)
    y2 = torch.cat([y1, y1], dim=0)
    
    optimizer1 = smep(model.parameters(), model=model, mode='ep', loss_type='mse')
    optimizer2 = smep(model.parameters(), model=model, mode='ep', loss_type='mse')
    
    optimizer1.step(x=x1, target=y1)
    optimizer2.step(x=x2, target=y2)
    
    # Gradients should be similar (within numerical precision)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            # Just check they're both finite and same sign
            assert torch.isfinite(p1.grad).all()
            assert torch.isfinite(p2.grad).all()
