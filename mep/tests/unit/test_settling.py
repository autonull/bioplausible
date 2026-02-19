"""
Tests for settling dynamics, including adaptive early stopping.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers.settling import Settler
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector

def test_settle_adaptive_stopping(device):
    """Test that settler stops early when energy converges."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 2)
    ).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    x = torch.randn(4, 10, device=device)

    # Set a very large max_steps but reasonable tolerance
    max_steps = 1000
    tol = 1e-3
    patience = 5

    settler = Settler(
        steps=max_steps,
        lr=0.05,
        tol=tol,
        patience=patience
    )

    call_count = 0
    original_call = energy_fn.__call__

    def mocked_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(*args, **kwargs)

    # Settle
    settler.settle(model, x, target=None, beta=0.0, energy_fn=mocked_call, structure=structure)

    assert call_count < max_steps, f"Settler should have stopped early (steps: {call_count} vs max: {max_steps})"
    assert call_count > patience, f"Settler should run at least patience steps (steps: {call_count})"


def test_settle_adaptive_step_size(device):
    """Test that adaptive step size logic works (does not crash and converges)."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 2)
    ).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    x = torch.randn(4, 10, device=device)

    # Use adaptive step size
    settler = Settler(
        steps=50,
        lr=0.5, # High LR to trigger rejection
        adaptive=True
    )

    # We want to check if it converges without diverging (NaN)
    # High LR without adaptive would likely oscillate or diverge

    states = settler.settle(model, x, target=None, beta=0.0, energy_fn=energy_fn, structure=structure)

    assert len(states) == 2
    for s in states:
        assert not torch.isnan(s).any()
        assert not torch.isinf(s).any()

def test_settle_divergence(device):
    """Test that settler raises RuntimeError on divergence (NaN/Inf)."""
    model = nn.Sequential(
        nn.Linear(10, 10)
    ).to(device)

    # Create an energy function that returns NaN
    def bad_energy_fn(*args, **kwargs):
        return torch.tensor(float('nan'), device=device)

    inspector = ModelInspector()
    structure = inspector.inspect(model)
    x = torch.randn(4, 10, device=device)

    settler = Settler(steps=10)

    with pytest.raises(RuntimeError, match="Energy diverged"):
        settler.settle(model, x, target=None, beta=0.0, energy_fn=bad_energy_fn, structure=structure)

def test_settle_with_graph_adaptive(device):
    """Test adaptive stopping for settle_with_graph."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 2)
    ).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    x = torch.randn(4, 10, device=device)

    max_steps = 1000
    tol = 1e-3
    patience = 5

    settler = Settler(
        steps=max_steps,
        lr=0.05,
        tol=tol,
        patience=patience
    )

    call_count = 0
    original_call = energy_fn.__call__

    def mocked_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(*args, **kwargs)

    settled_states = settler.settle_with_graph(
        model, x, target=None, beta=0.0, energy_fn=mocked_call, structure=structure
    )

    assert call_count < max_steps, f"Settler should have stopped early (steps: {call_count})"
    for s in settled_states:
        assert not s.requires_grad

def test_settle_validation():
    """Test validation of Settler arguments."""
    from mep.optimizers.settling import Settler
    with pytest.raises(ValueError, match="Steps must be positive"):
        Settler(steps=0)
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        Settler(lr=0.0)
    with pytest.raises(ValueError, match="Tolerance must be non-negative"):
        Settler(tol=-1)
    with pytest.raises(ValueError, match="Patience must be non-negative"):
        Settler(patience=-1)

def test_settle_with_graph_warning():
    """Test warning when using adaptive with graph."""
    from mep.optimizers.settling import Settler
    settler = Settler(adaptive=True, steps=1) # 1 step is enough
    # Mock model and inputs
    model = nn.Linear(2, 2)
    x = torch.randn(2, 2)
    # Mock structure
    structure = [{"type": "layer", "module": model}]
    # Mock energy_fn that returns a value depending on states (so it has grad)
    # states is the 3rd argument to energy_fn
    def energy_fn(model, x, states, *args):
        return sum(s.sum() for s in states)

    with pytest.warns(UserWarning, match="Adaptive settling is not supported"):
        settler.settle_with_graph(model, x, None, 0.0, energy_fn, structure)
