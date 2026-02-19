"""
Tests for EP monitoring utilities.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers.monitor import (
    EPMonitor,
    EnergyMetrics,
    EpochMetrics,
)


@pytest.fixture
def simple_model(device):
    """Simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)


@pytest.fixture
def sample_data(device):
    """Sample input/output data."""
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 5, device=device)
    return x, y


class TestEnergyMetrics:
    """Tests for EnergyMetrics dataclass."""

    def test_energy_metrics_creation(self):
        """Test EnergyMetrics can be created."""
        metrics = EnergyMetrics(
            step=10,
            energy=1.5,
            energy_change=0.1,
            grad_norm=0.5,
            state_norm=2.0
        )
        assert metrics.step == 10
        assert metrics.energy == 1.5
        assert metrics.energy_change == 0.1

    def test_energy_metrics_defaults(self):
        """Test EnergyMetrics with default list."""
        metrics = EnergyMetrics(
            step=0,
            energy=0.0,
            energy_change=0.0,
            grad_norm=0.0,
            state_norm=0.0
        )
        assert len(metrics.__dataclass_fields__) > 0


class TestEpochMetrics:
    """Tests for EpochMetrics dataclass."""

    def test_epoch_metrics_creation(self):
        """Test EpochMetrics can be created."""
        metrics = EpochMetrics(
            epoch=1,
            free_energy=1.0,
            nudged_energy=1.5,
            energy_gap=0.5,
            gradient_norm=0.3,
            weight_change=0.1,
            settling_steps=20
        )
        assert metrics.epoch == 1
        assert metrics.energy_gap == 0.5

    def test_epoch_metrics_with_history(self):
        """Test EpochMetrics with energy history."""
        history = [
            EnergyMetrics(step=i, energy=float(i), energy_change=0.1,
                         grad_norm=0.5, state_norm=1.0)
            for i in range(5)
        ]
        metrics = EpochMetrics(
            epoch=1,
            free_energy=1.0,
            nudged_energy=1.5,
            energy_gap=0.5,
            gradient_norm=0.3,
            weight_change=0.1,
            settling_steps=20,
            energy_history=history
        )
        assert len(metrics.energy_history) == 5


class TestEPMonitor:
    """Tests for EPMonitor class."""

    def test_monitor_initialization(self):
        """Test EPMonitor can be initialized."""
        monitor = EPMonitor()
        assert monitor is not None
        assert monitor.current_epoch == 0

    def test_monitor_start_epoch(self):
        """Test starting an epoch."""
        monitor = EPMonitor()
        monitor.start_epoch()
        assert monitor.current_epoch == 1

    def test_monitor_record_settling_step(self, device):
        """Test recording settling steps."""
        monitor = EPMonitor()
        monitor.start_epoch()
        
        states = [torch.randn(4, 20, device=device)]
        grads = [torch.randn(4, 20, device=device)]
        
        metrics = monitor.record_settling_step(
            step=0,
            energy=1.0,
            prev_energy=None,
            states=states,
            grads=grads
        )
        assert isinstance(metrics, EnergyMetrics)
        assert metrics.energy == 1.0
        assert len(monitor.settling_history) == 1

    def test_monitor_end_epoch(self, simple_model):
        """Test ending an epoch."""
        monitor = EPMonitor()
        from mep import muon_backprop
        optimizer = muon_backprop(simple_model.parameters(), lr=0.01)
        
        monitor.start_epoch()
        
        # Do a forward/backward pass to set gradients
        x = torch.randn(4, 10, device=next(simple_model.parameters()).device)
        y = torch.randn(4, 5, device=next(simple_model.parameters()).device)
        output = simple_model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        
        metrics = monitor.end_epoch(simple_model, optimizer)
        
        assert isinstance(metrics, EpochMetrics)
        assert metrics.epoch == 1

    def test_monitor_get_summary(self, simple_model):
        """Test getting summary statistics."""
        from mep import muon_backprop
        optimizer = muon_backprop(simple_model.parameters(), lr=0.01)
        monitor = EPMonitor()
        
        # Run a few epochs
        for epoch in range(3):
            monitor.start_epoch()
            x = torch.randn(4, 10, device=next(simple_model.parameters()).device)
            y = torch.randn(4, 5, device=next(simple_model.parameters()).device)
            output = simple_model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            monitor.end_epoch(simple_model, optimizer)
        
        assert monitor.current_epoch == 3
        assert len(monitor.epoch_metrics) == 3

    def test_monitor_reset(self, simple_model):
        """Test resetting the monitor."""
        from mep import muon_backprop
        optimizer = muon_backprop(simple_model.parameters(), lr=0.01)
        monitor = EPMonitor()
        
        monitor.start_epoch()
        x = torch.randn(4, 10, device=next(simple_model.parameters()).device)
        y = torch.randn(4, 5, device=next(simple_model.parameters()).device)
        output = simple_model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        monitor.end_epoch(simple_model, optimizer)
        
        # Reset
        monitor.current_epoch = 0
        monitor.epoch_metrics = []
        monitor.settling_history = []
        
        assert monitor.current_epoch == 0
        assert len(monitor.epoch_metrics) == 0

    def test_monitor_energy_gap_tracking(self, simple_model, device):
        """Test that energy gap is tracked correctly."""
        from mep import muon_backprop
        optimizer = muon_backprop(simple_model.parameters(), lr=0.01)
        monitor = EPMonitor()
        monitor.start_epoch()
        
        # Record settling steps
        states = [torch.randn(4, 20, device=device)]
        grads = [torch.randn(4, 20, device=device)]
        
        monitor.record_settling_step(
            step=0,
            energy=1.0,
            prev_energy=None,
            states=states,
            grads=grads
        )
        monitor.record_settling_step(
            step=1,
            energy=0.9,
            prev_energy=1.0,
            states=states,
            grads=grads
        )
        
        # Check history was recorded
        assert len(monitor.settling_history) == 2
        assert monitor.settling_history[0].energy == 1.0
        assert monitor.settling_history[1].energy == 0.9
        assert abs(monitor.settling_history[1].energy_change + 0.1) < 1e-6

    def test_monitor_check_convergence(self, device):
        """Test convergence detection."""
        monitor = EPMonitor()
        monitor.start_epoch()
        
        states = [torch.randn(4, 20, device=device)]
        grads = [torch.randn(4, 20, device=device)]
        
        # Record steps with decreasing energy change
        for i in range(10):
            monitor.record_settling_step(
                step=i,
                energy=1.0 - i * 0.01,
                prev_energy=1.0 - (i-1) * 0.01 if i > 0 else None,
                states=states,
                grads=grads
            )
        
        # Should converge with small changes
        assert monitor.check_convergence(tolerance=0.1, min_steps=5)

    def test_monitor_get_energy_trajectory(self, device):
        """Test energy trajectory retrieval."""
        monitor = EPMonitor()
        monitor.start_epoch()
        
        states = [torch.randn(4, 20, device=device)]
        grads = [torch.randn(4, 20, device=device)]
        
        energies = [1.0, 0.9, 0.8, 0.7]
        for i, e in enumerate(energies):
            monitor.record_settling_step(
                step=i,
                energy=e,
                prev_energy=energies[i-1] if i > 0 else None,
                states=states,
                grads=grads
            )
        
        trajectory = monitor.get_energy_trajectory()
        assert trajectory == energies

    def test_monitor_summary(self, simple_model):
        """Test summary generation."""
        from mep import muon_backprop
        optimizer = muon_backprop(simple_model.parameters(), lr=0.01)
        monitor = EPMonitor()
        
        # Empty summary
        assert "No epochs" in monitor.summary()
        
        # Add some epochs
        for epoch in range(3):
            monitor.start_epoch()
            x = torch.randn(4, 10, device=next(simple_model.parameters()).device)
            y = torch.randn(4, 5, device=next(simple_model.parameters()).device)
            output = simple_model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            monitor.end_epoch(simple_model, optimizer)
        
        summary = monitor.summary()
        assert "EP Training Summary" in summary
        assert "3 epochs" in summary
