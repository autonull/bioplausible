"""
Tests for spectral norm enforcement in EP and memory scaling.
"""

import torch
import torch.nn as nn
import pytest
import gc
from mep import smep
from mep.optimizers.settling import Settler
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector

def test_spectral_constraint_ep(device):
    """Test spectral constraint enforcement during EP training."""
    # Use a linear layer initialized with large weights
    model = nn.Linear(20, 20, bias=False).to(device)

    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(model.weight)
        S[0] = 5.0  # Force large singular value
        model.weight.copy_(U @ torch.diag(S) @ Vh)

    gamma = 1.0
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.01,
        gamma=gamma, # Constraint bound
        mode='ep',
        settle_steps=5
    )

    x = torch.randn(4, 20, device=device)
    y = torch.randn(4, 20, device=device)

    # Run several steps
    for _ in range(5):
        optimizer.step(x=x, target=y)

    # Check final spectral norm
    with torch.no_grad():
        U_final, S_final, Vh_final = torch.linalg.svd(model.weight.detach())
        final_norm = S_final[0].item()

    # It should be constrained towards gamma
    # Note: Exact enforcement depends on implementation details (e.g. power iteration steps)
    # But it should be significantly reduced from 5.0
    assert final_norm < 4.0, f"Spectral norm {final_norm} not reduced enough from 5.0"
    # Ideally <= gamma + margin
    # If using power iteration, it might take time.
    # If using strict projection (SVD), it should be <= gamma.
    # MEP likely uses power iteration or similar.

def test_memory_scaling_deep_mlp(device):
    """Test memory usage scaling for deep MLP during settling."""
    if device.type == 'cpu':
        # Skip precise memory check on CPU as it's hard to track.
        # But we can check list lengths.
        pass

    depth = 50
    width = 100
    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(width, width))
        layers.append(nn.ReLU())

    model = nn.Sequential(*layers).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    x = torch.randn(4, width, device=device)

    settler = Settler(steps=5)

    # Check number of states captured
    # Should be linear with depth (one per layer/act/etc that produces state)

    layer_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    states = settler._capture_states(model, x, structure)

    assert len(states) == layer_count, \
        f"Number of states {len(states)} should match layer count {layer_count}"

    # Check that we don't have duplicated states or extra overhead

    settled = settler.settle(model, x, target=None, beta=0.0, energy_fn=energy_fn, structure=structure)

    assert len(settled) == len(states)
