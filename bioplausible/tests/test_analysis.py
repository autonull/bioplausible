"""
Tests for Analysis Utilities
"""

import pytest
import torch
import torch.nn as nn

from bioplausible.analysis import DynamicsAnalyzer
from bioplausible.models.looped_mlp import LoopedMLP


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.gradient_method = "equilibrium"
        self.max_steps = 5

    def forward(self, x, steps=None, return_trajectory=False, return_dynamics=False):
        if return_dynamics:
            # Return mock dynamics
            return self.linear(x), {
                "trajectory": [x, x],
                "deltas": [0.1],
                "final_delta": 0.1,
            }
        return self.linear(x)


def test_dynamics_analyzer_init():
    model = MockModel()
    analyzer = DynamicsAnalyzer(model)
    assert analyzer.model == model


def test_get_convergence_data():
    model = MockModel()
    analyzer = DynamicsAnalyzer(model)
    x = torch.randn(2, 10)

    data = analyzer.get_convergence_data(x)
    assert "trajectory" in data
    assert "deltas" in data
    assert "fixed_point" in data


def test_gradient_alignment_restores_state():
    # Test that gradient_method is restored even if backward fails
    model = LoopedMLP(10, 10, 2)
    model.gradient_method = "equilibrium"  # Set initial state

    analyzer = DynamicsAnalyzer(model)
    x = torch.randn(2, 10)
    y = torch.randint(0, 2, (2,))

    # We can't easily force failure inside the function without mocking too much,
    # but we can verify it runs and restores.

    try:
        analyzer.compute_gradient_alignment(x, y)
    except Exception:
        pass

    assert model.gradient_method == "equilibrium"

    # Now verify it returns a value (LoopedMLP supports this)
    align = analyzer.compute_gradient_alignment(x, y)
    assert isinstance(align, float)
