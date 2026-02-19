
import torch
import torch.nn as nn
import pytest
from mep.presets import sdmep

def test_amp_compatibility(device):
    """
    Test that SDMEP runs under torch.amp.autocast.
    """
    if device.type == 'cpu':
        # Check for bfloat16 support on CPU
        try:
            torch.zeros(1, dtype=torch.bfloat16)
            amp_dtype = torch.bfloat16
        except RuntimeError:
            pytest.skip("CPU does not support bfloat16 for AMP testing")
    else:
        amp_dtype = torch.float16

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)

    optimizer = sdmep(model.parameters(), model=model, lr=0.01, beta=0.1)

    x = torch.randn(16, 32, device=device)
    y = torch.randint(0, 10, (16,), device=device)

    # Run step under autocast
    try:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            optimizer.zero_grad()
            optimizer.step(x=x, target=y)
    except Exception as e:
        pytest.fail(f"SDMEP step failed under AMP: {e}")

    # Check if gradients are computed and are finite
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(p.grad).any(), f"Gradient has NaN for {name}"
        assert not torch.isinf(p.grad).any(), f"Gradient has Inf for {name}"

def test_settling_precision_in_amp(device):
    """
    Test that settling maintains stability (no NaNs) under AMP.
    """
    if device.type == 'cpu':
        try:
            torch.zeros(1, dtype=torch.bfloat16)
            amp_dtype = torch.bfloat16
        except RuntimeError:
            pytest.skip("CPU does not support bfloat16")
    else:
        amp_dtype = torch.float16

    model = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(device)

    # Use larger beta/lr to stress test stability
    optimizer = sdmep(model.parameters(), model=model, lr=0.01, beta=0.5, settle_lr=0.1, settle_steps=20)

    x = torch.randn(32, 100, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
        try:
            optimizer.step(x=x, target=y)
        except RuntimeError as e:
            if "Energy diverged" in str(e):
                pytest.fail("Energy diverged under AMP")
            else:
                raise e
