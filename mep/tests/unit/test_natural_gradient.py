
import torch
import torch.nn as nn
import pytest
from mep.presets import natural_ep
from mep.optimizers.strategies.update import FisherUpdate

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)

def test_natural_gradient_fisher_creation():
    """Test that Fisher matrix is created and stored in state."""
    torch.manual_seed(42)
    model = SimpleModel()
    # natural_ep returns CompositeOptimizer
    optimizer = natural_ep(
        model.parameters(),
        model=model,
        beta=0.1,
        settle_steps=5,
        fisher_approx="empirical",
        use_diagonal_fisher=False
    )

    x = torch.randn(2, 4)
    y = torch.randn(2, 3)

    # Step
    optimizer.step(x=x, target=y)

    # Check if Fisher matrix exists in state
    fc_weight = model.fc.weight
    state = optimizer.state[fc_weight]

    assert "fisher" in state
    fisher = state["fisher"]
    # fc weight is (3, 4). Flattened gradient is (3, 4).
    # Fisher is g.T @ g -> (4, 4)
    assert fisher.shape == (4, 4)

    # Check if it's PSD?
    # g.T @ g is always PSD.

    # Check values are not zero (assuming non-zero gradient)
    assert fisher.abs().sum() > 0

def test_natural_gradient_diagonal_fisher():
    """Test diagonal Fisher approximation."""
    torch.manual_seed(42)
    model = SimpleModel()
    optimizer = natural_ep(
        model.parameters(),
        model=model,
        beta=0.1,
        settle_steps=5,
        fisher_approx="empirical",
        use_diagonal_fisher=True
    )

    x = torch.randn(2, 4)
    y = torch.randn(2, 3)

    optimizer.step(x=x, target=y)

    fc_weight = model.fc.weight
    state = optimizer.state[fc_weight]

    assert "fisher" in state
    fisher = state["fisher"]
    # Diagonal Fisher is (In,) -> (4,)
    assert fisher.shape == (4,)

def test_natural_gradient_conv2d():
    """Test Natural Gradient with Conv2d layers (ND tensor flattening)."""
    torch.manual_seed(42)
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            # In=2, Out=4, K=3. Weights (4, 2, 3, 3)
            self.conv = nn.Conv2d(2, 4, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    model = ConvModel()
    optimizer = natural_ep(
        model.parameters(),
        model=model,
        beta=0.1,
        settle_steps=5,
        fisher_approx="empirical",
        use_diagonal_fisher=False
    )

    x = torch.randn(2, 2, 5, 5)
    y = torch.randn(2, 4, 3, 3) # Output of 3x3 kernel on 5x5 is 3x3

    optimizer.step(x=x, target=y)

    conv_weight = model.conv.weight
    state = optimizer.state[conv_weight]

    assert "fisher" in state
    fisher = state["fisher"]

    # Weight shape (4, 2, 3, 3). Numel = 4 * 18 = 72.
    # Flattened shape (Out, In_features) = (4, 18).
    # Fisher shape (In_features, In_features) = (18, 18).
    assert fisher.shape == (18, 18)

def test_fisher_update_moving_average():
    """Test that Fisher matrix is updated with moving average."""
    # We can test FisherUpdate logic directly by mocking state/param
    param = nn.Parameter(torch.randn(3, 4))
    grad = torch.randn(3, 4)
    fisher_estimate = grad.T @ grad

    param.fisher = fisher_estimate.clone()

    state = {}

    update_strategy = FisherUpdate(beta=0.5, use_diagonal=False)

    # First step
    # Should initialize state['fisher'] = fisher_estimate
    update_strategy.transform_gradient(param, grad, state, {})

    assert not hasattr(param, 'fisher') # Consumed
    assert "fisher" in state
    assert torch.allclose(state["fisher"], fisher_estimate)

    # Second step
    grad2 = torch.randn(3, 4)
    fisher_estimate_2 = grad2.T @ grad2
    param.fisher = fisher_estimate_2.clone()

    prev_fisher = state["fisher"].clone()

    update_strategy.transform_gradient(param, grad2, state, {})

    # Expected: 0.5 * prev + (1 - 0.5) * new = 0.5 * prev + 0.5 * new
    expected = 0.5 * prev_fisher + 0.5 * fisher_estimate_2

    assert torch.allclose(state["fisher"], expected)
