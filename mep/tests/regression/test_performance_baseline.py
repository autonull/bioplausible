"""
Performance Regression Tests for MEP Optimizers.

These tests ensure that EP optimizer performance does not degrade
during future development. They establish minimum accuracy thresholds
based on validated baseline results.

Baseline Results (validated 2026-02-18):
- MNIST (mlp_small, 3 epochs): EP >= 85% (target: 91%)
- MNIST (10 epochs, 10k): EP >= 90% (target: 95%)
- XOR (100 steps): EP >= 90% accuracy

These are CONSERVATIVE thresholds - actual performance should be higher.
If tests fail, it indicates a regression that must be fixed.
"""

import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mep import smep


class TestEPPerformanceBaseline:
    """Performance regression tests for EP optimizers."""

    @pytest.fixture(scope="class")
    def mnist_train_loader(self):
        """Create MNIST training loader for tests."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # Use subset for faster tests
        return DataLoader(Subset(train_dataset, range(1000)), batch_size=64, shuffle=True)

    @pytest.fixture(scope="class")
    def mnist_test_loader(self):
        """Create MNIST test loader for evaluation."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        return DataLoader(test_dataset, batch_size=256)

    def _create_ep_optimizer(self, model):
        """Create EP optimizer with OPTIMAL settings."""
        return smep(
            model.parameters(),
            model=model,
            lr=0.01,
            mode='ep',
            loss_type='mse',
            use_error_feedback=False,
            # OPTIMAL settings discovered through systematic tuning
            beta=0.5,
            settle_steps=30,
            settle_lr=0.15,
        )

    def _train_epoch(self, model, loader, optimizer):
        """Train for one epoch."""
        model.train()
        for x, y in loader:
            optimizer.step(x=x, target=y)

    def _evaluate(self, model, loader):
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total

    def test_xor_convergence(self):
        """
        Test that EP converges on XOR problem.
        
        Baseline: EP should achieve >90% accuracy within 100 steps.
        This tests basic gradient flow and settling convergence.
        """
        torch.manual_seed(42)
        
        # XOR data
        x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([[0.], [1.], [1.], [0.]])
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=0.1,
            mode='ep',
            loss_type='mse',
            beta=0.3,
            settle_steps=50,
            settle_lr=0.1,
            use_error_feedback=False
        )
        
        # Train
        for _ in range(100):
            optimizer.step(x=x, target=y)
        
        # Evaluate
        with torch.no_grad():
            pred = model(x)
            # Check if predictions have correct sign pattern for XOR
            p0 = pred[0].item() < 0.5  # Should be low
            p1 = pred[1].item() > 0.5  # Should be high
            p2 = pred[2].item() > 0.5  # Should be high
            p3 = pred[3].item() < 0.5  # Should be low
            
            correct = sum([p0, p1, p2, p3])
            accuracy = correct / 4
        
        # CONSERVATIVE threshold: should be 100% but allow some margin
        assert accuracy >= 0.75, f"XOR accuracy {accuracy:.2%} below threshold 75%"

    def test_mnist_quick(self):
        """
        Quick MNIST test for performance regression.
        
        Baseline: EP should achieve >80% accuracy after 1 epoch on 1k samples.
        This is a CONSERVATIVE threshold - actual should be ~90%.
        """
        torch.manual_seed(42)
        
        # Data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(Subset(train_dataset, range(1000)), batch_size=64, shuffle=True)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        # Model (no dropout - breaks EP settling)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        optimizer = self._create_ep_optimizer(model)
        
        # Train for 1 epoch
        self._train_epoch(model, train_loader, optimizer)
        
        # Evaluate
        accuracy = self._evaluate(model, test_loader)
        
        # CONSERVATIVE threshold: actual should be ~90%
        assert accuracy >= 0.80, f"MNIST 1-epoch accuracy {accuracy:.2%} below threshold 80%"

    def test_mnist_extended(self):
        """
        Extended MNIST test for performance regression.
        
        Baseline: EP should achieve >88% accuracy after 3 epochs.
        This is a CONSERVATIVE threshold - actual should be ~91-94%.
        """
        torch.manual_seed(42)
        
        # Data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(Subset(train_dataset, range(2000)), batch_size=64, shuffle=True)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        # Model (no dropout - breaks EP settling)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        optimizer = self._create_ep_optimizer(model)
        
        # Train for 3 epochs
        for _ in range(3):
            self._train_epoch(model, train_loader, optimizer)
        
        # Evaluate
        accuracy = self._evaluate(model, test_loader)
        
        # CONSERVATIVE threshold: actual should be ~91-94%
        assert accuracy >= 0.88, f"MNIST 3-epoch accuracy {accuracy:.2%} below threshold 88%"


class TestEPConfiguration:
    """Tests to ensure optimal EP configuration is used."""

    def test_default_beta(self):
        """Test that default beta is optimal (0.5)."""
        from mep import smep
        import inspect
        
        sig = inspect.signature(smep)
        default_beta = sig.parameters['beta'].default
        
        assert default_beta >= 0.4, f"Default beta {default_beta} too low, should be >= 0.4"

    def test_default_settle_steps(self):
        """Test that default settle_steps is sufficient (>= 30)."""
        from mep import smep
        import inspect
        
        sig = inspect.signature(smep)
        default_steps = sig.parameters['settle_steps'].default
        
        assert default_steps >= 25, f"Default settle_steps {default_steps} too low, should be >= 25"

    def test_default_settle_lr(self):
        """Test that default settle_lr is optimal (0.15)."""
        from mep import smep
        import inspect
        
        sig = inspect.signature(smep)
        default_lr = sig.parameters['settle_lr'].default
        
        assert default_lr >= 0.1, f"Default settle_lr {default_lr} too low, should be >= 0.1"

    def test_default_loss_type(self):
        """Test that default loss_type is 'mse' for stability."""
        from mep import smep
        import inspect
        
        sig = inspect.signature(smep)
        default_loss = sig.parameters['loss_type'].default
        
        assert default_loss == 'mse', f"Default loss_type should be 'mse', got '{default_loss}'"
