"""
Tests for spectral norm constraints.
"""

import torch
import pytest
from mep import smep, sdmep
from mep.optimizers.strategies.constraint import (
    NoConstraint,
    SpectralConstraint,
    SettlingSpectralPenalty,
)


def test_spectral_constraint_scaling(device):
    """Test that spectral constraint correctly scales down large norms."""
    model = torch.nn.Linear(20, 20, bias=False).to(device)

    # Initialize with large weights
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(model.weight)
        S[0] = 10.0  # Force large singular value
        model.weight.copy_(U @ torch.diag(S) @ Vh)

    gamma = 0.95
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.1,
        gamma=gamma,
        mode='backprop'
    )

    # Run steps to enforce constraint
    for _ in range(10):
        x = torch.randn(4, 20, device=device)
        y = torch.randn(4, 20, device=device)
        output = model(x)
        loss = torch.nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

    # Check final spectral norm
    U_final, S_final, Vh_final = torch.linalg.svd(model.weight.detach())
    final_norm = S_final[0].item()

    # Should be close to gamma or less
    assert final_norm <= gamma + 0.05, f"Spectral norm {final_norm} > {gamma + 0.05}"


def test_spectral_constraint_no_change_if_small(device):
    """Test that spectral constraint does not affect small norms."""
    model = torch.nn.Linear(20, 20, bias=False).to(device)

    # Initialize with small weights
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(model.weight)
        S = S * 0.1  # Scale down
        model.weight.copy_(U @ torch.diag(S) @ Vh)

    orig_weight = model.weight.detach().clone()

    gamma = 1.0  # Larger than any SV
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.1,
        gamma=gamma,
        mode='backprop',
        weight_decay=0.0,
        momentum=0.0
    )

    # Zero gradients - no update should happen
    for _ in range(5):
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()

    # Should be unchanged (except maybe numerical noise)
    assert torch.allclose(model.weight, orig_weight, atol=1e-5)


class TestNoConstraint:
    """Tests for NoConstraint strategy."""

    def test_no_constraint_pass_through(self):
        """Test that NoConstraint doesn't modify parameters."""
        constraint = NoConstraint()
        param = torch.nn.Parameter(torch.randn(10, 10))
        state = {}
        group_config = {}

        original = param.data.clone()
        constraint.enforce(param, state, group_config)

        assert torch.allclose(param.data, original)


class TestSpectralConstraint:
    """Tests for SpectralConstraint strategy."""

    def test_spectral_constraint_init(self):
        """Test SpectralConstraint initialization."""
        constraint = SpectralConstraint(gamma=0.9, power_iter=5, timing='post_update')
        assert constraint.gamma == 0.9
        assert constraint.power_iter == 5
        assert constraint.timing == 'post_update'

    def test_spectral_constraint_invalid_gamma(self):
        """Test that invalid gamma raises error."""
        with pytest.raises(ValueError, match="gamma must be in"):
            SpectralConstraint(gamma=1.5)
        with pytest.raises(ValueError, match="gamma must be in"):
            SpectralConstraint(gamma=0)

    def test_spectral_constraint_invalid_timing(self):
        """Test that invalid timing raises error."""
        with pytest.raises(ValueError, match="Spectral timing"):
            SpectralConstraint(timing='invalid')

    def test_spectral_constraint_should_apply(self):
        """Test should_apply method for different timings."""
        constraint_post = SpectralConstraint(timing='post_update')
        constraint_settling = SpectralConstraint(timing='during_settling')
        constraint_both = SpectralConstraint(timing='both')

        assert constraint_post.should_apply('post_update')
        assert not constraint_post.should_apply('during_settling')

        assert not constraint_settling.should_apply('post_update')
        assert constraint_settling.should_apply('during_settling')

        assert constraint_both.should_apply('post_update')
        assert constraint_both.should_apply('during_settling')

    def test_spectral_constraint_1d_param(self):
        """Test that 1D parameters (biases) are skipped."""
        constraint = SpectralConstraint()
        param = torch.nn.Parameter(torch.randn(10))  # 1D bias
        state = {}

        original = param.data.clone()
        constraint.enforce(param, state, {})

        assert torch.allclose(param.data, original)

    def test_spectral_constraint_power_iteration(self):
        """Test power iteration method."""
        constraint = SpectralConstraint(gamma=0.95, power_iter=10)
        W = torch.randn(20, 20) * 0.1  # Small random matrix

        sigma, u, v = constraint._power_iteration(W, None, None)

        assert sigma > 0
        assert u.shape == (20,)
        assert v.shape == (20,)
        assert torch.isfinite(sigma)

    def test_spectral_constraint_conv2d(self):
        """Test spectral constraint on Conv2d weights."""
        constraint = SpectralConstraint(gamma=0.95)
        param = torch.nn.Parameter(torch.randn(32, 3, 3, 3))  # Conv2d weight
        state = {}

        # Should not raise error
        constraint.enforce(param, state, {})

        # Check state was updated
        assert 'u_spec' in state
        assert 'v_spec' in state


class TestSettlingSpectralPenalty:
    """Tests for SettlingSpectralPenalty strategy."""

    def test_penalty_init(self):
        """Test SettlingSpectralPenalty initialization."""
        penalty = SettlingSpectralPenalty(gamma=0.9, lambda_penalty=2.0)
        assert penalty.gamma == 0.9
        assert penalty.lambda_penalty == 2.0

    def test_penalty_zero_for_small_weights(self):
        """Test that penalty is zero for small weights."""
        penalty = SettlingSpectralPenalty(gamma=1.0, lambda_penalty=1.0)

        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        with torch.no_grad():
            model[0].weight.mul_(0.1)  # Small weights

        optimizer_state = {}
        result = penalty.compute_penalty(model, optimizer_state)

        assert result.item() == 0.0

    def test_penalty_positive_for_large_weights(self):
        """Test that penalty is positive for large weights."""
        penalty = SettlingSpectralPenalty(gamma=0.1, lambda_penalty=1.0)

        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        with torch.no_grad():
            model[0].weight.mul_(10.0)  # Large weights

        optimizer_state = {}
        result = penalty.compute_penalty(model, optimizer_state)

        assert result.item() > 0

    def test_penalty_1d_param_skipped(self):
        """Test that 1D parameters are skipped."""
        penalty = SettlingSpectralPenalty(gamma=0.1)

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        # Add a bias (1D param)
        optimizer_state = {}
        result = penalty.compute_penalty(model, optimizer_state)

        # Should not raise error
        assert torch.isfinite(result)
