
import torch
import torch.nn as nn
import pytest
from mep.presets import local_ep

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def test_local_ep_cnn_runs():
    torch.manual_seed(42)
    device = "cpu"
    model = SimpleCNN().to(device)
    optimizer = local_ep(
        model.parameters(),
        model=model,
        lr=0.01,
        settle_steps=5
    )

    x = torch.randn(2, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (2,), device=device)

    # Run a step
    optimizer.step(x=x, target=y)

    # Check if parameters changed
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        # We expect gradients to be non-zero generally, but with random init and small steps/beta it might be small.
        # But LocalEP should produce gradients for all layers involved.

def test_local_ep_last_layer_bn():
    """Test LocalEP where the last learnable module is NOT a 'layer' type."""
    # Example: Conv -> BN -> ReLU. No Linear at the end.
    # This checks if BN params are updated.

    class ConvBNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            # Output shape (N, 4, H, W).
            # Target should match.
            return x

    torch.manual_seed(42)
    model = ConvBNNet()
    optimizer = local_ep(
        model.parameters(),
        model=model,
        lr=0.01,
        settle_steps=5
    )

    x = torch.randn(2, 1, 8, 8)
    # Target same shape as output
    y = torch.randn(2, 4, 8, 8)

    optimizer.step(x=x, target=y)

    # Check gradients
    assert model.conv.weight.grad is not None

    # BN params should ideally have gradients if they affect the energy.
    # But in LocalEP logic, BN params are updated with the *next* layer.
    # Here there is no next layer.
    # So we expect BN params to NOT have gradients (current bug/limitation).

    has_grad = model.bn.weight.grad is not None
    print(f"BN weight grad present: {has_grad}")

    # This test asserts the DESIRED behavior (that they SHOULD have gradients).
    # If it fails, it confirms the bug.
    assert model.bn.weight.grad is not None, "BN weight should have gradient even if it's after the last layer"
