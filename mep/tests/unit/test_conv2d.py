"""
Tests for Conv2d support in EP optimizers.

Verifies that convolutional layers work correctly with:
- State capture during settling
- Energy computation with 4D tensors
- EP gradient computation
"""

import torch
import torch.nn as nn
import pytest
from mep import smep, sdmep


class SimpleCNN(nn.Module):
    """Simple CNN for testing Conv2d support."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


@pytest.fixture
def cnn_model(device):
    """Create a simple CNN model."""
    return SimpleCNN(num_classes=10).to(device)


@pytest.fixture
def cnn_data(device):
    """Create sample CNN input data (MNIST-like)."""
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    return x, y


def test_cnn_backprop(device, cnn_model, cnn_data):
    """Test CNN with backprop mode."""
    x, y = cnn_data

    optimizer = smep(
        cnn_model.parameters(),
        model=cnn_model,
        lr=0.01,
        mode='backprop'
    )

    output = cnn_model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()


def test_cnn_ep(device, cnn_model, cnn_data):
    """Test CNN with EP mode."""
    x, y = cnn_data

    optimizer = smep(
        cnn_model.parameters(),
        model=cnn_model,
        lr=0.01,
        mode='ep',
        settle_steps=5
    )

    optimizer.step(x=x, target=y)


def test_cnn_sdmep(device, cnn_model, cnn_data):
    """Test CNN with SDMEP."""
    x, y = cnn_data

    optimizer = sdmep(
        cnn_model.parameters(),
        model=cnn_model,
        lr=0.01,
        settle_steps=5
    )

    optimizer.step(x=x, target=y)


def test_cnn_training_loop(device, cnn_model):
    """Test CNN training loop."""
    x = torch.randn(8, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    optimizer = smep(
        cnn_model.parameters(),
        model=cnn_model,
        lr=0.01,
        mode='backprop'
    )

    initial_loss = None
    for i in range(10):
        output = cnn_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if initial_loss is None:
            initial_loss = loss.item()
    
    # Loss should be finite
    assert torch.isfinite(loss), f"Final loss should be finite, got {loss}"


def test_cnn_spectral_constraint(device, cnn_model, cnn_data):
    """Test CNN with spectral constraints."""
    x, y = cnn_data

    optimizer = smep(
        cnn_model.parameters(),
        model=cnn_model,
        lr=0.01,
        gamma=0.95,
        mode='backprop'
    )

    for _ in range(5):
        output = cnn_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
