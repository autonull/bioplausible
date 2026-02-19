"""
Benchmark regression tests for MEP optimizers.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytest
import torch.nn as nn
from mep import sdmep, smep, muon_backprop


def _train_and_evaluate(model, optimizer, train_loader, test_loader, device, use_ep=False):
    """Helper to train model and return accuracy."""
    # Train 1 epoch
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)

        if use_ep:
            optimizer.step(x=x, target=y)
        else:
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def _create_model_and_loaders(device):
    """Create model and data loaders for MNIST test."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    except Exception as e:
        pytest.skip(f"Could not download MNIST: {e}")

    # Create subsets for speed
    train_indices = range(500)
    test_indices = range(100)
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)

    # Model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    return model, train_loader, test_loader


@pytest.mark.slow
def test_mnist_backprop(device):
    """Test MNIST with backprop mode (Muon optimizer)."""
    model, train_loader, test_loader = _create_model_and_loaders(device)

    optimizer = muon_backprop(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
    )

    accuracy = _train_and_evaluate(model, optimizer, train_loader, test_loader, device, use_ep=False)

    # Should be better than random (10%)
    assert accuracy > 0.15, f"Backprop accuracy too low: {accuracy}"


@pytest.mark.slow
def test_mnist_smep_backprop_mode(device):
    """Test MNIST with SMEP in backprop mode."""
    model, train_loader, test_loader = _create_model_and_loaders(device)

    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.01,
        momentum=0.9,
        mode='backprop',
    )

    accuracy = _train_and_evaluate(model, optimizer, train_loader, test_loader, device, use_ep=False)

    # Should be better than random (10%)
    assert accuracy > 0.15, f"SMEP backprop accuracy too low: {accuracy}"


@pytest.mark.slow
def test_mnist_smep_ep_mode(device):
    """Test MNIST with SMEP in EP mode.
    
    Note: EP requires more training iterations than backprop.
    This test verifies EP runs correctly and produces gradients,
    not that it achieves high accuracy in 1 epoch.
    """
    model, train_loader, test_loader = _create_model_and_loaders(device)

    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.01,
        momentum=0.9,
        mode='ep',
        settle_steps=10,
        settle_lr=0.05,
        loss_type='cross_entropy'
    )

    # Verify EP runs without errors and produces gradients
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x = x.view(x.size(0), -1)
    
    optimizer.step(x=x, target=y)
    
    # Check that gradients were computed
    has_grads = all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads, "EP did not produce gradients"
    
    # Verify model can make predictions (no NaN/Inf)
    model.eval()
    with torch.no_grad():
        output = model(x)
        assert torch.isfinite(output).all(), "Model produced NaN/Inf outputs"


@pytest.mark.slow
def test_mnist_sdmep_ep_mode(device):
    """Test MNIST with SDMEP in EP mode.
    
    Note: EP requires more training iterations than backprop.
    This test verifies EP runs correctly and produces gradients,
    not that it achieves high accuracy in 1 epoch.
    """
    model, train_loader, test_loader = _create_model_and_loaders(device)

    optimizer = sdmep(
        model.parameters(),
        model=model,
        lr=0.01,
        momentum=0.9,
        mode='ep',
        settle_steps=10,
        settle_lr=0.05,
        loss_type='cross_entropy'
    )

    # Verify EP runs without errors and produces gradients
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x = x.view(x.size(0), -1)
    
    optimizer.step(x=x, target=y)
    
    # Check that gradients were computed
    has_grads = all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads, "SDMEP did not produce gradients"
    
    # Verify model can make predictions (no NaN/Inf)
    model.eval()
    with torch.no_grad():
        output = model(x)
        assert torch.isfinite(output).all(), "Model produced NaN/Inf outputs"
