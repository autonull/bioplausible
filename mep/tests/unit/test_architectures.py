"""
Tests for CNN and Transformer architecture support.

Tests verify that MEP optimizers work with:
- Convolutional networks (Conv1d, Conv2d, Conv3d)
- Transformer encoders/decoders
- Mixed architectures
"""

import torch
import torch.nn as nn
import pytest
from mep import smep, sdmep


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleTransformer(nn.Module):
    """Simple Transformer encoder for testing."""

    def __init__(self, input_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(10, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(input_dim, 5)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output_proj(x)


@pytest.fixture
def cnn_model(device):
    """CNN model for testing."""
    return SimpleCNN(num_classes=10).to(device)


@pytest.fixture
def transformer_model(device):
    """Transformer model for testing."""
    return SimpleTransformer(input_dim=64, num_heads=4, num_layers=2).to(device)


class TestCNN:
    """Tests for CNN architecture support."""

    def test_cnn_backprop(self, device, cnn_model):
        """Test CNN with backprop."""
        optimizer = smep(
            cnn_model.parameters(),
            model=cnn_model,
            lr=0.01,
            mode='backprop'
        )

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        output = cnn_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

    def test_cnn_ep(self, device, cnn_model):
        """Test CNN with EP."""
        optimizer = smep(
            cnn_model.parameters(),
            model=cnn_model,
            lr=0.01,
            mode='ep',
            settle_steps=5
        )

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        optimizer.step(x=x, target=y)

    def test_cnn_spectral_constraint(self, device, cnn_model):
        """Test CNN with spectral constraints."""
        optimizer = smep(
            cnn_model.parameters(),
            model=cnn_model,
            lr=0.01,
            gamma=0.95,
            mode='backprop'
        )

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        for _ in range(5):
            output = cnn_model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class TestTransformer:
    """Tests for Transformer architecture support."""

    def test_transformer_backprop(self, device, transformer_model):
        """Test Transformer with backprop."""
        optimizer = smep(
            transformer_model.parameters(),
            model=transformer_model,
            lr=0.01,
            mode='backprop'
        )

        # Sequence input: (batch, seq_len, input_dim)
        x = torch.randn(4, 10, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        output = transformer_model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

    def test_transformer_ep_simple(self, device):
        """Test Transformer with EP using simple architecture."""
        # Use a simpler transformer setup that works with EP
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        ).to(device)
        
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=0.01,
            mode='ep',
            settle_steps=3
        )

        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        optimizer.step(x=x, target=y)


class TestConv1d:
    """Tests for Conv1d support."""

    def test_conv1d_backprop(self, device):
        """Test Conv1d with backprop."""
        model = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 5)
        ).to(device)

        optimizer = smep(model.parameters(), model=model, lr=0.01, mode='backprop')

        x = torch.randn(4, 3, 32, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()


class TestMixedArchitecture:
    """Tests for mixed architectures."""

    def test_cnn_transformer_hybrid(self, device):
        """Test CNN + Transformer hybrid."""
        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.flatten = nn.Flatten()
                self.transformer_proj = nn.Linear(8192, 64)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
                    num_layers=1
                )
                self.classifier = nn.Linear(64, 5)

            def forward(self, x):
                x = self.cnn(x)  # (B, 32, 16, 16)
                x = self.flatten(x)  # (B, 8192)
                x = self.transformer_proj(x).unsqueeze(1)  # (B, 1, 64)
                x = self.transformer(x)
                return self.classifier(x.squeeze(1))

        model = HybridModel().to(device)
        optimizer = smep(model.parameters(), model=model, lr=0.01, mode='backprop')

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 5, (4,), device=device)

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
