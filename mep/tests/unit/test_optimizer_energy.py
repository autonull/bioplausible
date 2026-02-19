"""
Tests for energy computation and settling dynamics.
"""

import torch
import torch.nn as nn
import pytest
from mep import smep
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector
from mep.optimizers.settling import Settler


def test_settle_energy_reduction(device):
    """Test that settling reduces energy compared to initial state."""
    # Simple MLP
    dims = [10, 20, 5]
    model = nn.Sequential(
        nn.Linear(dims[0], dims[1]),
        nn.ReLU(),
        nn.Linear(dims[1], dims[2])
    ).to(device)

    energy_fn = EnergyFunction()
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    batch_size = 4
    x = torch.randn(batch_size, dims[0]).to(device)
    
    # Capture initial states
    # We can use Settler._capture_states logic manually or instantiate Settler
    # But for test control, let's use Settler.settle directly
    
    # Let's instantiate Settler to use helper
    settler = Settler(steps=20, lr=0.05)
    initial_states = settler._capture_states(model, x, structure)
    
    E_initial = energy_fn(model, x, initial_states, structure, target_vec=None, beta=0.0)
    
    # Settle
    settled_states = settler.settle(model, x, target=None, beta=0.0, energy_fn=energy_fn, structure=structure)
    
    # Compute settled energy
    E_settled = energy_fn(model, x, settled_states, structure, target_vec=None, beta=0.0)
    
    # Energy should decrease
    assert E_settled.item() <= E_initial.item() + 1e-5, \
        f"Settling should reduce energy: Initial {E_initial.item()}, Settled {E_settled.item()}"


def test_energy_with_nudge(device):
    """Test energy computation with nudging."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    ).to(device)
    
    energy_fn = EnergyFunction(loss_type='mse')
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)
    target_vec = nn.functional.one_hot(y, num_classes=2).float()
    
    settler = Settler()
    states = settler._capture_states(model, x, structure)
    
    # Energy without nudge
    E_free = energy_fn(model, x, states, structure, target_vec=None, beta=0.0)
    
    # Energy with nudge
    E_nudged = energy_fn(model, x, states, structure, target_vec=target_vec, beta=0.5)
    
    # Nudged energy should be different (usually higher because of extra term, unless states perfectly match target)
    assert E_nudged.item() != E_free.item(), "Nudge should change energy"


def test_classification_energy(device):
    """Test energy computation for classification."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    ).to(device)
    
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 3, (4,), device=device)
    
    settler = Settler()
    states = settler._capture_states(model, x, structure)
    
    # Energy with classification target
    E = energy_fn(model, x, states, structure, target_vec=y, beta=0.5)
    
    # Should be finite
    assert torch.isfinite(E), f"Energy should be finite, got {E}"


def test_energy_complex_layers(device):
    """Test energy computation with BatchNorm and Attention."""
    # Note: MultiheadAttention by default expects (seq, batch, feature)
    embed_dim = 8
    model = nn.Sequential(
        nn.Linear(10, embed_dim),
        nn.BatchNorm1d(embed_dim),
        nn.Linear(embed_dim, embed_dim),
    ).to(device)

    # Manually add attention to structure for testing energy function logic
    attention = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True).to(device)

    inspector = ModelInspector()
    structure = inspector.inspect(model)

    # Append attention to structure manually
    structure.append({"type": "attention", "module": attention})
    # Add a final layer to produce state
    final_layer = nn.Linear(embed_dim, 2).to(device)
    structure.append({"type": "layer", "module": final_layer})

    energy_fn = EnergyFunction(loss_type='mse')

    x = torch.randn(4, 10, device=device) # (batch, feature)

    # We need states for:
    # 1. Linear(10, 8) -> state 0
    # 2. Linear(8, 8) -> state 1
    # 3. Attention -> state 2
    # 4. Final Linear -> state 3

    # Let's generate dummy states
    states = [
        torch.randn(4, 8, device=device),
        torch.randn(4, 8, device=device),
        torch.randn(4, 8, device=device),
        torch.randn(4, 2, device=device)
    ]

    # Run energy computation
    E = energy_fn(model, x, states, structure, target_vec=None, beta=0.0)

    assert torch.isfinite(E)
