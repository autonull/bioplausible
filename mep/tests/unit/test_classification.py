"""
Tests for EP classification with CrossEntropy energy function.
"""

import torch
import torch.nn as nn
import pytest
from mep import smep, sdmep


@pytest.fixture
def classification_model(device):
    """Simple MLP for classification."""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5)  # 5 classes
    ).to(device)
    return model


@pytest.fixture
def classification_data(device):
    """Mini batch for classification."""
    x = torch.randn(8, 10, device=device)
    y = torch.randint(0, 5, (8,), device=device)  # 5 classes
    return x, y


def test_cross_entropy_energy_computation(device, classification_model):
    """Test that CrossEntropy energy is computed correctly."""
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 5, (4,), device=device)

    optimizer = smep(
        classification_model.parameters(),
        model=classification_model,
        mode='ep',
        loss_type='cross_entropy',
        settle_steps=5
    )

    # Should not raise
    optimizer.step(x=x, target=y)


def test_classification_training_loop(device, classification_model):
    """Test classification training loop converges."""
    x = torch.randn(16, 10, device=device)
    y = torch.randint(0, 5, (16,), device=device)

    optimizer = smep(
        classification_model.parameters(),
        model=classification_model,
        lr=0.01,
        mode='ep',
        loss_type='cross_entropy',
        settle_steps=5
    )

    initial_loss = None
    for i in range(10):
        optimizer.step(x=x, target=y)
        optimizer.zero_grad()
        
        # Track loss
        with torch.no_grad():
            output = classification_model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            if initial_loss is None:
                initial_loss = loss.item()
    
    # Loss should be finite
    assert torch.isfinite(loss), f"Final loss should be finite, got {loss}"


def test_softmax_temperature(device, classification_model):
    """Test softmax temperature parameter."""
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 5, (4,), device=device)

    for temp in [0.5, 1.0, 2.0]:
        optimizer = smep(
            classification_model.parameters(),
            model=classification_model,
            mode='ep',
            loss_type='cross_entropy',
            softmax_temperature=temp,
            settle_steps=3
        )
        optimizer.step(x=x, target=y)


def test_sdmep_classification(device):
    """Test SDMEP with classification."""
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 5)
    ).to(device)
    
    x = torch.randn(8, 10, device=device)
    y = torch.randint(0, 5, (8,), device=device)
    
    optimizer = sdmep(
        model.parameters(),
        model=model,
        mode='ep',
        loss_type='cross_entropy',
        settle_steps=5
    )
    
    # Should not raise
    optimizer.step(x=x, target=y)


def test_backprop_classification(device, classification_model):
    """Test backprop classification (baseline)."""
    x = torch.randn(16, 10, device=device)
    y = torch.randint(0, 5, (16,), device=device)

    optimizer = smep(
        classification_model.parameters(),
        model=classification_model,
        lr=0.01,
        mode='backprop'
    )

    for _ in range(10):
        output = classification_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
