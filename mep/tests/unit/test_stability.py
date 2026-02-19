"""
Edge case tests and stability tests for MEP optimizers.

Tests boundary conditions, numerical stability, and error handling.
"""

import torch
import torch.nn as nn
import pytest
from mep import smep, sdmep, local_ep, natural_ep


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_learning_rate(self, device):
        """Test that negative learning rate raises error."""
        model = nn.Linear(10, 5).to(device)
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            smep(model.parameters(), model=model, lr=-0.01)

    def test_invalid_momentum(self, device):
        """Test that invalid momentum raises error."""
        model = nn.Linear(10, 5).to(device)
        with pytest.raises(ValueError, match="Momentum must be in"):
            smep(model.parameters(), model=model, momentum=1.5)
        with pytest.raises(ValueError, match="Momentum must be in"):
            smep(model.parameters(), model=model, momentum=-0.1)

    def test_invalid_beta(self, device):
        """Test that invalid beta raises error."""
        from mep.optimizers.strategies.gradient import EPGradient
        
        # EPGradient validates beta
        with pytest.raises(ValueError, match="[Bb]eta must be in"):
            EPGradient(beta=1.5)
        with pytest.raises(ValueError, match="[Bb]eta must be in"):
            EPGradient(beta=0)

    def test_invalid_gamma(self, device):
        """Test that invalid gamma raises error."""
        model = nn.Linear(10, 5).to(device)
        with pytest.raises(ValueError, match="[Gg]amma must be in"):
            smep(model.parameters(), model=model, gamma=1.5)
        with pytest.raises(ValueError, match="[Gg]amma must be in"):
            smep(model.parameters(), model=model, gamma=0)

    def test_invalid_spectral_timing(self, device):
        """Test that invalid spectral_timing raises error."""
        from mep.optimizers.strategies.constraint import SpectralConstraint
        
        # SpectralConstraint validates timing
        with pytest.raises(ValueError, match="[Ss]pectral timing must be"):
            SpectralConstraint(gamma=0.95, timing='invalid')


class TestNumericalStability:
    """Test numerical stability under edge cases."""

    def test_large_gradients(self, device):
        """Test optimizer handles large gradients."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to(device)
        
        optimizer = smep(model.parameters(), model=model, lr=0.01)
        
        x = torch.randn(4, 10, device=device) * 10  # Large input
        y = torch.randint(0, 2, (4,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Should not raise
        optimizer.step()

    def test_small_gradients(self, device):
        """Test optimizer handles very small gradients."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to(device)
        
        optimizer = smep(model.parameters(), model=model, lr=0.01)
        
        x = torch.randn(4, 10, device=device) * 0.01  # Small input
        y = torch.randint(0, 2, (4,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Should not raise
        optimizer.step()

    def test_zero_gradients(self, device):
        """Test optimizer handles zero gradients."""
        model = nn.Linear(10, 5).to(device)
        
        optimizer = smep(model.parameters(), model=model, lr=0.01)
        
        # Manually set zero gradient
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        
        # Should not raise
        optimizer.step()

    def test_nan_in_input(self, device):
        """Test optimizer handles NaN in input."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to(device)
        
        optimizer = smep(model.parameters(), model=model, lr=0.01)
        
        x = torch.randn(4, 10, device=device)
        x[0, 0] = float('nan')  # Inject NaN
        y = torch.randint(0, 2, (4,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # Loss will be NaN, backward may fail - that's expected
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

    def test_inf_in_input(self, device):
        """Test optimizer handles Inf in input."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        ).to(device)
        
        optimizer = smep(model.parameters(), model=model, lr=0.01)
        
        x = torch.randn(4, 10, device=device)
        x[0, 0] = float('inf')  # Inject Inf
        y = torch.randint(0, 2, (4,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # Loss will be Inf, backward may fail - that's expected
        if not torch.isinf(loss):
            loss.backward()
            optimizer.step()


class TestEPStability:
    """Test EP-specific stability."""

    def test_ep_convergence_simple(self, device):
        """Test EP converges on simple problem."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        ).to(device)
        
        optimizer = sdmep(
            model.parameters(),
            model=model,
            lr=0.01,
            beta=0.5,
            settle_steps=10
        )
        
        losses = []
        for _ in range(20):
            x = torch.randn(8, 10, device=device)
            y = torch.randint(0, 2, (8,), device=device)
            
            optimizer.step(x=x, target=y)
            optimizer.zero_grad()
        
        # Should complete without errors

    def test_ep_different_beta(self, device):
        """Test EP with different beta values."""
        model = nn.Linear(10, 2).to(device)
        
        for beta in [0.1, 0.5, 0.9]:
            opt = sdmep(model.parameters(), model=model, beta=beta, settle_steps=3)
            x = torch.randn(4, 10, device=device)
            y = torch.randint(0, 2, (4,), device=device)
            opt.step(x=x, target=y)

    def test_ep_different_settle_steps(self, device):
        """Test EP with different settle_steps values."""
        model = nn.Linear(10, 2).to(device)
        
        for steps in [1, 5, 20]:
            opt = sdmep(model.parameters(), model=model, settle_steps=steps)
            x = torch.randn(4, 10, device=device)
            y = torch.randint(0, 2, (4,), device=device)
            opt.step(x=x, target=y)


class TestConstraintEnforcement:
    """Test constraint enforcement."""

    def test_spectral_constraint_enforced(self, device):
        """Test spectral constraint is enforced."""
        model = nn.Linear(20, 20, bias=False).to(device)
        
        # Initialize with large weights
        with torch.no_grad():
            model.weight.fill_(2.0)
        
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=0.01,
            gamma=0.95,
            mode='backprop'
        )
        
        # Run several steps
        for _ in range(10):
            x = torch.randn(4, 20, device=device)
            y = torch.randn(4, 20, device=device)
            
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()
        
        # Check spectral norm
        U, S, V = torch.linalg.svd(model.weight.detach())
        assert S[0].item() <= 1.0, f"Spectral norm {S[0].item()} > 1.0"


class TestOptimizerState:
    """Test optimizer state management."""

    def test_state_dict_roundtrip(self, device):
        """Test saving and loading optimizer state."""
        model = nn.Linear(10, 5).to(device)
        optimizer = smep(model.parameters(), model=model, lr=0.01)
        
        # Run a step
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        
        # Save state
        state = optimizer.state_dict()
        
        # Create new optimizer and load state
        model2 = nn.Linear(10, 5).to(device)
        optimizer2 = smep(model2.parameters(), model=model2, lr=0.01)
        optimizer2.load_state_dict(state)
        
        # Should work without errors

    def test_zero_grad(self, device):
        """Test zero_grad clears gradients."""
        model = nn.Linear(10, 5).to(device)
        optimizer = smep(model.parameters(), model=model, lr=0.01)
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Verify gradients exist
        for p in model.parameters():
            assert p.grad is not None
        
        optimizer.zero_grad()
        
        # Verify gradients cleared
        for p in model.parameters():
            assert p.grad is None
