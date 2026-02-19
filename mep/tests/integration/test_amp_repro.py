
import torch
import torch.nn as nn
import pytest
from mep.optimizers.energy import EnergyFunction
from mep.presets import smep
from mep.optimizers.settling import Settler

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_amp_energy_accumulation_dtype():
    """Test that EnergyFunction accumulates in float32 even with BF16/FP16 inputs."""
    device = "cpu"
    dtype = torch.bfloat16 # Supported on most modern CPUs

    try:
        x = torch.randn(2, 10, device=device, dtype=dtype)
    except RuntimeError:
        pytest.skip("BFloat16 not supported on this CPU")

    model = SimpleMLP().to(device) # FP32 weights

    # Mock states as FP32 (as Settler should capture them)
    states = [
        torch.randn(2, 20, device=device, dtype=torch.float32),
        torch.randn(2, 1, device=device, dtype=torch.float32)
    ]

    structure = [
        {"type": "layer", "module": model.fc1},
        {"type": "act", "module": model.relu},
        {"type": "layer", "module": model.fc2}
    ]

    energy_fn = EnergyFunction()

    # x is BF16. States are FP32.
    # We must run in autocast so that linear(bf16, fp32) works (or behaves as expected in AMP)
    # On CPU, autocast handles bf16 inputs with fp32 weights.
    with torch.autocast(device_type="cpu", dtype=dtype):
        # Energy function should return FP32 energy.
        E = energy_fn(model, x, states, structure)

    assert E.dtype == torch.float32, f"Energy should be float32, got {E.dtype}"
    assert torch.isfinite(E)

def test_settler_captures_fp32():
    """Test that Settler captures states in FP32 even if forward pass is low precision."""
    device = "cpu"
    dtype = torch.bfloat16
    try:
        _ = torch.tensor([1.0], dtype=dtype)
    except RuntimeError:
        pytest.skip("BFloat16 not supported")

    model = SimpleMLP().to(device)
    x = torch.randn(2, 10, device=device, dtype=dtype)

    settler = Settler(steps=1)

    # Mock structure
    from mep.optimizers.inspector import ModelInspector
    inspector = ModelInspector()
    structure = inspector.inspect(model)

    energy_fn = EnergyFunction()

    # With autocast to simulate AMP forward pass
    with torch.autocast(device_type="cpu", dtype=dtype):
        # Even if autocast is on, capture should return FP32
        states = settler.settle(model, x, None, 0.0, energy_fn, structure)

    for s in states:
        assert s.dtype == torch.float32, f"State should be float32, got {s.dtype}"
        # settle returns detached states, so requires_grad is False.
        # But during settling they required grad. The fact that we got results means it worked.

def test_optimizer_step_with_autocast():
    """Test optimizer.step() with autocast enabled (integration test)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for full AMP test")

    device = "cuda"
    model = SimpleMLP().to(device)
    optimizer = smep(model.parameters(), model=model, mode='ep', settle_steps=5)

    x = torch.randn(16, 10, device=device)
    y = torch.randn(16, 1, device=device)

    # Run with autocast
    with torch.cuda.amp.autocast():
        # This should run:
        # 1. Forward pass (autocasted)
        # 2. State capture (casted to FP32)
        # 3. Settling (FP32 states, FP16 forward via prev.to(x.dtype))
        # 4. Contrast (FP32)
        optimizer.step(x=x, target=y)

    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

if __name__ == "__main__":
    test_amp_energy_accumulation_dtype()
    test_settler_captures_fp32()
    print("Tests passed!")
