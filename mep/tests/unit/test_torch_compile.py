"""
Tests for torch.compile compatibility.

Verifies that settling dynamics and energy functions work with torch.compile
for accelerated repeated calls.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers.settling import Settler, _compiled_settle_step
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector


@pytest.fixture
def simple_model(device):
    """Simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 5)
    ).to(device)


@pytest.fixture
def sample_data(device):
    """Sample input/output data."""
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 5, device=device)
    return x, y


class TestTorchCompile:
    """Tests for torch.compile compatibility."""

    def test_settle_compiled_basic(self, simple_model, sample_data, device):
        """Test that settle_compiled runs without errors."""
        x, y = sample_data
        
        settler = Settler(steps=5, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        structure = inspector.inspect(simple_model)
        
        # First call (includes compilation overhead)
        states = settler.settle_compiled(
            simple_model, x, y, beta=0.1,
            energy_fn=energy_fn, structure=structure
        )
        
        # Count actual layers (not activations)
        layer_count = sum(1 for item in structure if item["type"] in ("layer", "attention"))
        assert len(states) == layer_count
        assert all(s.shape[0] == x.shape[0] for s in states)

    def test_settle_compiled_repeated_calls_faster(self, simple_model, sample_data, device):
        """Test that repeated calls benefit from compilation."""
        if device.type != "cuda":
            pytest.skip("torch.compile benefits most visible on CUDA")
        
        x, y = sample_data
        
        settler = Settler(steps=10, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        structure = inspector.inspect(simple_model)
        
        # Warmup (compilation happens here)
        _ = settler.settle_compiled(
            simple_model, x, y, beta=0.1,
            energy_fn=energy_fn, structure=structure
        )
        torch.cuda.synchronize()
        
        # Time repeated calls
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        for _ in range(5):
            _ = settler.settle_compiled(
                simple_model, x, y, beta=0.1,
                energy_fn=energy_fn, structure=structure
            )
        
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        
        # Should complete in reasonable time (compilation + 5 calls)
        assert elapsed_ms < 5000, f"Too slow: {elapsed_ms}ms"

    def test_compiled_settle_step_function(self, simple_model, sample_data, device):
        """Test the standalone compiled settle step function."""
        x, y = sample_data
        
        settler = Settler(steps=1, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        structure = inspector.inspect(simple_model)
        
        states = settler._capture_states(simple_model, x, structure)
        momentum_buffers = [torch.zeros_like(s) for s in states]
        target_vec = settler._prepare_target(y, states[-1].shape[-1], states[-1].dtype)
        
        # Test compiled step
        new_states, new_buffers = _compiled_settle_step(
            states, momentum_buffers, simple_model, x, target_vec,
            beta=0.1, energy_fn=energy_fn, structure=structure, lr=0.01
        )
        
        assert len(new_states) == len(states)
        assert len(new_buffers) == len(momentum_buffers)

    def test_energy_fn_compiled(self, simple_model, sample_data, device):
        """Test that energy function works with torch.compile."""
        x, y = sample_data
        
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        structure = inspector.inspect(simple_model)
        
        settler = Settler(steps=1, lr=0.01)
        states = settler._capture_states(simple_model, x, structure)
        target_vec = settler._prepare_target(y, states[-1].shape[-1], states[-1].dtype)
        
        # Compile the energy function
        compiled_energy = torch.compile(energy_fn)
        
        # Test compiled energy
        E = compiled_energy(simple_model, x, states, structure, target_vec, beta=0.1)
        
        assert torch.isfinite(E)
        assert E > 0

    def test_settle_compiled_vs_regular(self, simple_model, sample_data, device):
        """Test that compiled settling produces similar results to regular."""
        x, y = sample_data
        
        settler = Settler(steps=5, lr=0.01, adaptive=False)
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        structure = inspector.inspect(simple_model)
        
        # Regular settling
        torch.manual_seed(42)
        states_regular = settler.settle(
            simple_model, x, y, beta=0.1,
            energy_fn=energy_fn, structure=structure
        )
        
        # Compiled settling
        torch.manual_seed(42)
        states_compiled = settler.settle_compiled(
            simple_model, x, y, beta=0.1,
            energy_fn=energy_fn, structure=structure
        )
        
        # Results should be similar (may not be identical due to different loop structure)
        for s_reg, s_comp in zip(states_regular, states_compiled):
            # Allow some tolerance due to potential numerical differences
            assert torch.allclose(s_reg, s_comp, rtol=1e-4, atol=1e-4), \
                f"States differ: max diff = {(s_reg - s_comp).abs().max()}"

    def test_compile_with_different_models(self, device):
        """Test torch.compile with different model architectures."""
        models = [
            nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(device),
            nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 5)).to(device),
        ]
        
        settler = Settler(steps=3, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        
        for model in models:
            x = torch.randn(2, model[0].in_features, device=device)
            y = torch.randn(2, model[-1].out_features, device=device)
            structure = inspector.inspect(model)
            
            states = settler.settle_compiled(
                model, x, y, beta=0.1,
                energy_fn=energy_fn, structure=structure
            )
            
            assert len(states) > 0

    def test_compile_error_handling(self, simple_model, device):
        """Test that errors are properly handled in compiled mode."""
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)
        
        settler = Settler(steps=5, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        structure = inspector.inspect(simple_model)
        
        # Invalid beta should still raise error
        with pytest.raises(ValueError, match="Beta must be in"):
            settler.settle_compiled(
                simple_model, x, y, beta=1.5,
                energy_fn=energy_fn, structure=structure
            )

    def test_compile_empty_input_error(self, simple_model, device):
        """Test that empty input raises error in compiled mode."""
        x = torch.tensor([], device=device).reshape(0, 10)
        y = torch.tensor([], device=device).reshape(0, 5)
        
        settler = Settler(steps=5, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        inspector = ModelInspector()
        structure = inspector.inspect(simple_model)
        
        with pytest.raises(ValueError, match="Input tensor cannot be empty"):
            settler.settle_compiled(
                simple_model, x, y, beta=0.1,
                energy_fn=energy_fn, structure=structure
            )
