"""
Tests for robustness of the EP pipeline.
Check input validation, NaN handling, and error messages.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.settling import Settler
from mep.optimizers.inspector import ModelInspector

def test_energy_function_validation(device):
    """Test validation in EnergyFunction."""
    model = nn.Sequential(nn.Linear(10, 5)).to(device)
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    energy_fn = EnergyFunction()

    x = torch.randn(4, 10, device=device)
    states = [torch.randn(4, 5, device=device)]

    # 1. Mismatch batch size (triggers shape mismatch first for MSE)
    with pytest.raises(ValueError, match="Shape mismatch"):
        energy_fn(model, x, states, structure, target_vec=torch.randn(3, 5, device=device), beta=0.1)

    # 2. Mismatch number of states
    with pytest.raises(ValueError, match="Number of states"):
        energy_fn(model, x, states + states, structure, target_vec=None)

    # 3. Shape mismatch in prediction vs state
    # Create states with wrong shape
    states_wrong = [torch.randn(4, 3, device=device)]
    with pytest.raises(ValueError, match="Shape mismatch"):
        energy_fn(model, x, states_wrong, structure)

def test_settler_invalid_inputs(device):
    """Test validation in Settler."""
    model = nn.Sequential(nn.Linear(10, 5)).to(device)
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    energy_fn = EnergyFunction()

    settler = Settler()

    # 1. Empty input
    with pytest.raises(ValueError, match="Input tensor cannot be empty"):
        settler.settle(model, torch.tensor([], device=device), None, 0.0, energy_fn, structure)

    # 2. Invalid beta
    with pytest.raises(ValueError, match="Beta"):
        settler.settle(model, torch.randn(4, 10, device=device), None, 1.5, energy_fn, structure)

def test_nan_divergence(device):
    """Test that NaN in energy raises RuntimeError."""
    model = nn.Sequential(nn.Linear(10, 5)).to(device)
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    x = torch.randn(4, 10, device=device)

    # Inject NaN into model weight to cause NaN energy
    with torch.no_grad():
        model[0].weight[0, 0] = float('nan')

    settler = Settler(steps=5)
    energy_fn = EnergyFunction()

    # NaN in weights causes RuntimeError during forward pass
    with pytest.raises(RuntimeError):
        settler.settle(model, x, None, 0.0, energy_fn, structure)
