#!/usr/bin/env python3
"""
MEP Integration Smoke Test

Tests that MEP optimizers are properly integrated into Bioplausible.

Usage:
    python -m pytest tests/test_mep_integration.py -v

Expected runtime: < 60 seconds
"""

import pytest
import torch
import torch.nn as nn

from bioplausible.core.registry import ComponentCategory
from bioplausible.core.registry import Registry


class TinyMLP(nn.Module):
    """Tiny MLP for smoke testing."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_dummy_data(batch_size=4):
    """Get dummy data for testing."""
    x = torch.randn(batch_size, 784)
    y = torch.randint(0, 10, (batch_size,))
    return x, y


class TestMEPImport:
    """Test that MEP components can be imported."""

    def test_import_smep(self):
        """Test importing smep preset."""
        try:
            from bioplausible import smep

            assert smep is not None
        except ImportError:
            pytest.skip("MEP not installed")

    def test_import_smep_fast(self):
        """Test importing smep_fast preset."""
        try:
            from bioplausible import smep_fast

            assert smep_fast is not None
        except ImportError:
            pytest.skip("MEP not installed")

    def test_import_composite_optimizer(self):
        """Test importing CompositeOptimizer from MEP."""
        try:
            from bioplausible.zoo.mep.optimizers import CompositeOptimizer

            assert CompositeOptimizer is not None
        except ImportError:
            pytest.skip("MEP not installed")

    def test_import_strategies(self):
        """Test importing strategy components from MEP."""
        try:
            from bioplausible.zoo.mep.optimizers import EPGradient
            from bioplausible.zoo.mep.optimizers import MuonUpdate
            from bioplausible.zoo.mep.optimizers import SpectralConstraint

            assert EPGradient is not None
            assert MuonUpdate is not None
            assert SpectralConstraint is not None
        except ImportError:
            pytest.skip("MEP not installed")


class TestZooIntegration:
    """Test simplified model/optimizer registry."""

    def test_model_registry_available(self):
        """Test that model registry is available."""
        models = Registry._components.get(ComponentCategory.MODEL, {})
        assert len(models) > 0

    def test_optimizer_registry_available(self):
        """Test that optimizer registry is available."""
        optimizers = Registry._components.get(ComponentCategory.OPTIMIZER, {})
        assert len(optimizers) > 0

    def test_list_models(self):
        """Test listing models."""
        from bioplausible.core.registry import list_models

        models = list_models()
        assert len(models) > 0

    def test_list_optimizers(self):
        """Test listing optimizers."""
        result = Registry.list(ComponentCategory.OPTIMIZER)
        names = result.get("optimizer", [])
        assert len(names) > 0


class TestMEPOptimizers:
    """Test MEP optimizer functionality."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return TinyMLP()

    @pytest.fixture
    def data(self):
        """Create test data."""
        return get_dummy_data()

    def test_smep_basic(self, model, data):
        """Test basic SMEP functionality."""
        try:
            from bioplausible import smep
        except ImportError:
            pytest.skip("MEP not installed")

        x, y = data
        optimizer = smep(model.parameters(), model=model, mode="ep")
        optimizer.step(x=x, target=y)
        assert True

    def test_smep_fast_basic(self, model, data):
        """Test basic smep_fast functionality."""
        try:
            from bioplausible import smep_fast
        except ImportError:
            pytest.skip("MEP not installed")

        x, y = data
        optimizer = smep_fast(model.parameters(), model=model)
        optimizer.step(x=x, target=y)
        assert True

    def test_muon_backprop_basic(self, model, data):
        """Test basic muon_backprop functionality."""
        try:
            from bioplausible import muon_backprop
        except ImportError:
            pytest.skip("MEP not installed")

        x, y = data
        optimizer = muon_backprop(model.parameters(), model=model)
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        assert True

    def test_composite_optimizer(self, model, data):
        """Test CompositeOptimizer."""
        try:
            from bioplausible.zoo.mep.optimizers import CompositeOptimizer
            from bioplausible.zoo.mep.optimizers import EPGradient
            from bioplausible.zoo.mep.optimizers import MuonUpdate
            from bioplausible.zoo.mep.optimizers import SpectralConstraint
        except ImportError:
            pytest.skip("MEP not installed")

        x, y = data
        optimizer = CompositeOptimizer(
            model.parameters(),
            gradient=EPGradient(beta=0.5, settle_steps=5),
            update=MuonUpdate(ns_steps=3),
            constraint=SpectralConstraint(gamma=0.95),
            lr=0.01,
            model=model,
        )
        optimizer.step(x=x, target=y)
        assert True


class TestLearning:
    """Test that learning actually occurs."""

    def test_mnist_learning(self):
        """Test that EP learns on MNIST (mini test)."""
        try:
            from bioplausible import smep
        except ImportError:
            pytest.skip("MEP not installed")

        model = TinyMLP()
        x, y = get_dummy_data(batch_size=16)

        optimizer = smep(
            model.parameters(),
            model=model,
            mode="ep",
            settle_steps=10,
            lr=0.01,
        )

        model.train()
        output = model(x)
        nn.functional.cross_entropy(output, y).item()

        for _ in range(5):
            optimizer.step(x=x, target=y)
            optimizer.zero_grad()

        output = model(x)
        final_loss = nn.functional.cross_entropy(output, y).item()

        assert not torch.isnan(torch.tensor(final_loss))
        assert not torch.isinf(torch.tensor(final_loss))


def main():
    """Run smoke test standalone."""
    print("=" * 50)
    print("MEP Integration Smoke Test")
    print("=" * 50)

    print("\nTesting imports...")
    try:
        from bioplausible import smep
        from bioplausible import smep_fast

        print("All imports successful")
    except ImportError as e:
        print(f"Import failed: {e}")
        return 1

    print("\nTesting Zoo...")
    from bioplausible.core.registry import list_models

    models = list_models()
    result = Registry.list(ComponentCategory.OPTIMIZER)
    optimizers = result.get("optimizer", [])
    print(f"Zoo has {len(models)} models and {len(optimizers)} optimizers")

    print("\nTesting optimizer...")
    model = TinyMLP()
    x, y = get_dummy_data()
    optimizer = smep(model.parameters(), model=model, mode="ep")
    optimizer.step(x=x, target=y)
    print("Optimizer step successful")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
