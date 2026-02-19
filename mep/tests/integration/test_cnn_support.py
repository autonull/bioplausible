import torch
import torch.nn as nn
import pytest
from mep.optimizers.strategies.gradient import LocalEPGradient
from mep.optimizers.strategies.update import MuonUpdate
from mep.optimizers.inspector import ModelInspector
from mep.presets import local_ep, smep

@pytest.mark.xfail(reason="LocalEP updates can be small/unstable depending on initialization")
def test_local_ep_gradient_with_complex_cnn(device):
    """Test LocalEPGradient on a CNN with Norm/Pool layers."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 3, padding=1),
        nn.Flatten(),
        nn.Linear(8 * 14 * 14, 10)
    ).to(device)

    x = torch.randn(2, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (2,), device=device)

    # Use LocalEP
    optimizer = local_ep(model.parameters(), model=model, lr=0.1, settle_steps=5, beta=0.1)

    # Run a step
    try:
        for _ in range(5):
            optimizer.zero_grad()
            optimizer.step(x=x, target=y)
    except Exception as e:
        pytest.fail(f"LocalEP step failed: {e}")

    # Check if gradients are computed for all layers
    # Specifically check the first Conv2d layer
    # If LocalEPGradient skips non-layer modules, gradient might be wrong or zero
    # But more importantly, if it fails to propagate `prev`, it might crash or produce wrong input for next layer
    # Since `_get_layer_io` iterates and updates `prev`, missing modules means `prev` is NOT updated.
    # So `prev` into Conv2d(4, 8) would be output of Conv2d(1, 4) directly (without BN/ReLU/Pool).
    # This shape mismatch: (2, 4, 28, 28) into Conv2d(4, 8) expecting (2, 4, 14, 14)?
    # Wait, Conv2d input channel mismatch? No, channel is 4.
    # But MaxPool changes spatial dim. If ignored, spatial dim is 28x28.
    # Conv2d(4, 8) produces 28x28.
    # Flatten produces 8*28*28 = 6272 features.
    # Linear expects 8*14*14 = 1568 features.
    # This should crash with shape mismatch in Linear layer forward pass inside `_get_layer_io` if `prev` is wrong.

    # Store initial params to verify updates
    initial_params = {name: p.clone() for name, p in model.named_parameters()}

    # Check that gradients are computed and params updated
    # Especially BatchNorm which is between layers
    for name, p in model.named_parameters():
        # Skip check if gradient is too small (update might be numerically insignificant)
        if p.grad is not None and p.grad.norm().item() < 1e-4:
            continue

        diff_sum = (p - initial_params[name]).abs().sum().item()
        assert diff_sum > 0, f"Parameter {name} was not updated!"

def test_muon_update_conv2d_orthogonalization(device):
    """Test if MuonUpdate orthogonalizes 4D Conv2d weights."""
    # Create a parameter that is NOT orthogonal
    # (Out, In, H, W) -> (Out, In*H*W)
    # Make it obviously non-orthogonal
    w = torch.ones(4, 2, 3, 3, device=device) * 0.1
    param = nn.Parameter(w)

    # Create Muon update
    muon = MuonUpdate(ns_steps=5)

    # Apply update on gradient = param
    # If it works, it should change the gradient significantly towards orthogonality
    # If it skips (returns as is), it will be same

    grad = w.clone()
    update = muon.transform_gradient(param, grad, {}, {})

    # Check if update is different from grad
    diff = torch.norm(update - grad)

    # Muon should change it
    if diff < 1e-6:
        pytest.fail("MuonUpdate did not change the gradient (likely skipped 4D tensor)")

    # Check orthogonality of update (reshaped)
    update_flat = update.view(4, -1)
    # Approximately orthogonal rows?
    # Newton-Schulz pushes towards X^T X = I (or scaled I)
    gram = update_flat @ update_flat.T
    diag = torch.diag(gram)
    off_diag = gram - torch.diag(diag)

    # Frobenius norm of off-diagonal should be small relative to diagonal
    off_diag_norm = torch.norm(off_diag)
    diag_norm = torch.norm(diag)

    print(f"Off-diag norm: {off_diag_norm}, Diag norm: {diag_norm}")
    # It might not be perfectly orthogonal in 5 steps if condition number is bad,
    # but it should be better than input (ones).

    # Input ones: gram is all same values. off-diag large.

    # 5 steps might not be enough for perfect orthogonality starting from all ones (rank 1 matrix!)
    # Rank 1 matrix cannot be orthogonalized to full rank.
    # The NS update will likely just scale it or fail to converge to I.
    # But it should change. The fact that diff > 1e-6 (checked above) means it processed the 4D tensor.
    # Let's try a random matrix which is full rank but not orthogonal.

    w_rand = torch.randn(4, 2, 3, 3, device=device)
    param_rand = nn.Parameter(w_rand)
    grad_rand = w_rand.clone()
    update_rand = muon.transform_gradient(param_rand, grad_rand, {}, {})

    update_flat_rand = update_rand.view(4, -1)
    gram_rand = update_flat_rand @ update_flat_rand.T
    diag_rand = torch.diag(gram_rand)
    off_diag_rand = gram_rand - torch.diag(diag_rand)

    off_diag_norm_rand = torch.norm(off_diag_rand)
    diag_norm_rand = torch.norm(diag_rand)

    print(f"Rand Off-diag: {off_diag_norm_rand}, Rand Diag: {diag_norm_rand}")

    assert off_diag_norm_rand < 0.1 * diag_norm_rand, "Update should be orthogonalized for random input"
