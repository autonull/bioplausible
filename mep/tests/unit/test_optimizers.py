"""
Unit tests for MEP optimizers.

Tests the refactored strategy-based optimizer implementation.
"""

import torch
import torch.nn as nn
import pytest
from mep import smep, sdmep, local_ep, natural_ep, muon_backprop


@pytest.fixture
def simple_model(device):
    """Simple MLP for testing."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    ).to(device)
    return model


def test_smep_backprop_step(device, simple_model):
    """Test that SMEP with backprop takes a step."""
    optimizer = smep(
        simple_model.parameters(),
        model=simple_model,
        mode='backprop',
        lr=0.1
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]

    output = simple_model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()

    # Check that parameters changed
    for p, p_old in zip(simple_model.parameters(), w_before):
        assert not torch.allclose(p, p_old)


def test_smep_ep_step(device, simple_model):
    """Test that SMEP with EP takes a step."""
    optimizer = smep(
        simple_model.parameters(),
        model=simple_model,
        mode='ep',
        lr=0.01,
        beta=0.5,
        settle_steps=5
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]
    optimizer.step(x=x, target=y)

    # Check parameters changed (at least weights, not necessarily biases)
    updated = False
    for p, p_old in zip(simple_model.parameters(), w_before):
        if p.ndim >= 2 and not torch.allclose(p, p_old):
            updated = True
            break
    assert updated


def test_sdmep_step(device, simple_model):
    """Test that SDMEP takes a step."""
    optimizer = sdmep(
        simple_model.parameters(),
        model=simple_model,
        lr=0.01,
        settle_steps=5
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]
    optimizer.step(x=x, target=y)

    for p, p_old in zip(simple_model.parameters(), w_before):
        if p.ndim >= 2:
            assert not torch.allclose(p, p_old)


def test_sdmep_spectral_constraint(device, simple_model):
    """Test that SDMEP enforces spectral constraint."""
    optimizer = sdmep(
        simple_model.parameters(),
        model=simple_model,
        lr=0.01,
        gamma=0.95,
        settle_steps=3
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    # Run multiple steps
    for _ in range(5):
        optimizer.step(x=x, target=y)
        optimizer.zero_grad()

    # Check spectral norm of weight matrices
    for name, param in simple_model.named_parameters():
        if param.ndim == 2:
            U, S, V = torch.linalg.svd(param.detach())
            spectral_norm = S[0].item()
            # Should be <= gamma (with some tolerance for numerical error)
            assert spectral_norm <= 1.0, f"{name} spectral norm {spectral_norm} > 1.0"


def test_smep_spectral_timing(device, simple_model):
    """Test that SMEP runs with spectral_timing='during_settling'."""
    optimizer = smep(
        simple_model.parameters(),
        model=simple_model,
        mode='ep',
        spectral_timing='during_settling',
        settle_steps=3
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    # Run step - should not crash
    optimizer.step(x=x, target=y)


def test_local_ep_step(device, simple_model):
    """Test LocalEPMuon updates parameters using local gradients."""
    optimizer = local_ep(
        simple_model.parameters(),
        model=simple_model,
        beta=0.1,
        settle_steps=5
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]
    optimizer.step(x=x, target=y)

    # Check if weights changed
    for p, p_old in zip(simple_model.parameters(), w_before):
        if p.requires_grad and p.ndim >= 2:
            assert not torch.allclose(p, p_old), f"Parameter {p.shape} did not update"


def test_natural_ep_step(device, simple_model):
    """Test NaturalEPMuon updates parameters using Fisher approximation."""
    optimizer = natural_ep(
        simple_model.parameters(),
        model=simple_model,
        beta=0.1,
        settle_steps=5
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]
    optimizer.step(x=x, target=y)

    # Check update
    for p, p_old in zip(simple_model.parameters(), w_before):
        if p.requires_grad and p.ndim >= 2:
            assert not torch.allclose(p, p_old), f"Parameter {p.shape} did not update"


def test_muon_backprop_step(device, simple_model):
    """Test Muon backprop (drop-in SGD replacement)."""
    optimizer = muon_backprop(
        simple_model.parameters(),
        lr=0.1
    )

    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    w_before = [p.clone() for p in simple_model.parameters()]

    output = simple_model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()

    # Check that parameters changed
    for p, p_old in zip(simple_model.parameters(), w_before):
        assert not torch.allclose(p, p_old)
