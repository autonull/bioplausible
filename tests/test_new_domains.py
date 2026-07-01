"""Tests for new domain interfaces (Tabular, TimeSeries, Scientific)."""

import pytest
import torch
import torch.nn as nn

from bioplausible.domains import (
    DomainType,
    TabularTask,
    TimeSeriesTask,
    ScientificTask,
    create_domain_task,
    list_domains,
)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class TestTabularTask:
    def test_domain_type(self):
        task = TabularTask(name="test", dataset_name="digits")
        assert task.domain_type == DomainType.TABULAR

    def test_spec(self):
        task = TabularTask(name="test", dataset_name="digits")
        spec = task.spec
        assert spec.domain_type == DomainType.TABULAR
        assert "tabular" in spec.tags

    def test_setup_digits(self):
        task = TabularTask(name="test", dataset_name="digits", batch_size=32)
        task.setup()
        assert task._input_dim == 64
        assert task._output_dim == 10
        assert task._setup_done

    def test_get_dataloader(self):
        task = TabularTask(name="test", dataset_name="digits", batch_size=32)
        loader = task.get_dataloader("train")
        assert loader is not None
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape[0] <= 32

    def test_evaluate(self):
        task = TabularTask(name="test", dataset_name="digits", batch_size=32)
        model = SimpleMLP(64, 10)
        metrics = task.evaluate(model, max_batches=2)
        assert metrics.loss >= 0
        assert metrics.accuracy is not None

    def test_unknown_dataset(self):
        task = TabularTask(name="test", dataset_name="nonexistent")
        with pytest.raises(ValueError):
            task.setup()


class TestTimeSeriesTask:
    def test_domain_type(self):
        task = TimeSeriesTask(name="test", dataset_name="synthetic")
        assert task.domain_type == DomainType.TIMESERIES

    def test_spec(self):
        task = TimeSeriesTask(name="test", dataset_name="synthetic")
        spec = task.spec
        assert "timeseries" in spec.tags
        assert spec.requires_sequence

    def test_setup_synthetic(self):
        task = TimeSeriesTask(
            name="test", dataset_name="synthetic",
            seq_len=32, horizon=1, batch_size=32
        )
        task.setup()
        assert task._input_dim == 32
        assert task._output_dim == 1

    def test_get_dataloader(self):
        task = TimeSeriesTask(
            name="test", dataset_name="synthetic",
            seq_len=16, horizon=1, batch_size=16
        )
        loader = task.get_dataloader("train")
        batch = next(iter(loader))
        assert batch[0].shape[-1] == 16

    def test_evaluate(self):
        task = TimeSeriesTask(
            name="test", dataset_name="synthetic",
            seq_len=16, horizon=1, batch_size=16
        )
        model = SimpleMLP(16, 1)
        metrics = task.evaluate(model, max_batches=2)
        assert "mse" in metrics.custom
        assert metrics.loss >= 0


class TestScientificTask:
    def test_domain_type(self):
        task = ScientificTask(name="test", dataset_name="pendulum")
        assert task.domain_type == DomainType.SCIENTIFIC

    def test_spec(self):
        task = ScientificTask(name="test", dataset_name="pendulum")
        spec = task.spec
        assert "scientific" in spec.tags

    def test_setup_pendulum(self):
        task = ScientificTask(name="test", dataset_name="pendulum", batch_size=32)
        task.setup()
        assert task._input_dim == 3
        assert task._output_dim == 3

    def test_setup_lorenz(self):
        task = ScientificTask(name="test", dataset_name="lorenz", batch_size=32)
        task.setup()
        assert task._input_dim == 3
        assert task._output_dim == 3

    def test_evaluate_pendulum(self):
        task = ScientificTask(name="test", dataset_name="pendulum", batch_size=32)
        model = SimpleMLP(3, 3)
        task.setup()
        metrics = task.evaluate(model, max_batches=2)
        assert metrics.loss >= 0

    def test_unknown_dataset(self):
        task = ScientificTask(name="test", dataset_name="nonexistent")
        with pytest.raises(ValueError):
            task.setup()


class TestDomainFactory:
    def test_create_domain_task(self):
        task = create_domain_task("tabular", "test", dataset_name="digits")
        assert isinstance(task, TabularTask)

        task = create_domain_task("timeseries", "test", dataset_name="synthetic")
        assert isinstance(task, TimeSeriesTask)

        task = create_domain_task("scientific", "test", dataset_name="pendulum")
        assert isinstance(task, ScientificTask)

    def test_list_domains(self):
        domains = list_domains()
        assert "tabular" in domains
        assert "timeseries" in domains
        assert "scientific" in domains
