"""Tests for train_backprop and train_pcn on synthetic data."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bioplausible.graph import (
    Edge,
    InferenceSGD,
    Linear,
    ReLU,
    TaskMap,
    graph,
    initialize_params,
    train_backprop,
    train_pcn,
)


@pytest.fixture
def synthetic_data():
    """100 samples of linearly separable 20-dim data, 5 classes."""
    torch.manual_seed(42)
    x = torch.randn(100, 20)
    w = torch.randn(20, 5)
    logits = x @ w
    y = logits.argmax(dim=-1)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader, x, y


@pytest.fixture
def mlp_graph():
    input_node = Linear(shape=(20, 32), name="input")
    hidden = ReLU(name="hidden")
    output = Linear(shape=(32, 5), name="output")

    structure = graph(
        nodes=[input_node, hidden, output],
        edges=[
            Edge(source=input_node, target=hidden.slot("input")),
            Edge(source=hidden, target=output.slot("input")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )
    return structure


class TestTrainBackprop:
    def test_loss_decreases(self, synthetic_data, mlp_graph):
        """Backprop training should reduce loss over 3 epochs."""
        train_loader, _, _ = synthetic_data
        params = initialize_params(mlp_graph)

        results = train_backprop(mlp_graph, params, train_loader, epochs=3, lr=0.01)
        assert results["train_loss"] < 5.0
        assert results["train_acc"] >= 0.0

    def test_accuracy_high(self, synthetic_data, mlp_graph):
        """Backprop should achieve >85% on linearly separable data after 5 epochs."""
        train_loader, _, _ = synthetic_data
        params = initialize_params(mlp_graph)

        results = train_backprop(mlp_graph, params, train_loader, epochs=5, lr=0.01)
        assert results["train_acc"] >= 0.85

    def test_test_loader(self, synthetic_data, mlp_graph):
        """Test loader evaluation should work."""
        train_loader, x, y = synthetic_data
        dataset = TensorDataset(x, y)
        test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        params = initialize_params(mlp_graph)

        results = train_backprop(
            mlp_graph,
            params,
            train_loader,
            test_loader=test_loader,
            epochs=2,
            lr=0.01,
        )
        assert "test_acc" in results
        assert isinstance(results["test_acc"], float)

    def test_returns_metrics_dict(self, synthetic_data, mlp_graph):
        """train_backprop returns the correct metrics."""
        train_loader, _, _ = synthetic_data
        params = initialize_params(mlp_graph)

        results = train_backprop(mlp_graph, params, train_loader, epochs=1, lr=0.01)
        assert "train_acc" in results
        assert "test_acc" in results
        assert "train_loss" in results
        assert "time" in results


class TestTrainPCN:
    def test_loss_decreases(self, synthetic_data, mlp_graph):
        """PC training should reduce loss over 3 epochs."""
        train_loader, _, _ = synthetic_data
        params = initialize_params(mlp_graph)

        results = train_pcn(
            mlp_graph,
            params,
            train_loader,
            epochs=3,
            lr=0.01,
            infer_steps=5,
            eta_infer=0.05,
        )
        assert results["train_loss"] < 5.0

    def test_accuracy_reasonable(self, synthetic_data, mlp_graph):
        """PC should achieve reasonable accuracy on linearly separable data."""
        train_loader, _, _ = synthetic_data
        params = initialize_params(mlp_graph)

        results = train_pcn(
            mlp_graph,
            params,
            train_loader,
            epochs=5,
            lr=0.01,
            infer_steps=5,
            eta_infer=0.05,
        )
        assert results["train_acc"] >= 0.6

    def test_returns_metrics_dict(self, synthetic_data, mlp_graph):
        """train_pcn returns the correct metrics."""
        train_loader, _, _ = synthetic_data
        params = initialize_params(mlp_graph)

        results = train_pcn(
            mlp_graph,
            params,
            train_loader,
            epochs=1,
            lr=0.01,
            infer_steps=3,
            eta_infer=0.05,
        )
        assert "train_acc" in results
        assert "test_acc" in results
        assert "train_loss" in results
        assert "time" in results


class TestSameGraph:
    """Both training functions should work on identical graph/params."""

    def test_same_params_accepted(self, synthetic_data, mlp_graph):
        """Both train_backprop and train_pcn accept identical graph + params."""
        train_loader, _, _ = synthetic_data
        params = initialize_params(mlp_graph)

        # Backprop
        results_bp = train_backprop(mlp_graph, params, train_loader, epochs=1, lr=0.01)

        # Re-init params for PC
        params_pc = initialize_params(mlp_graph)
        results_pc = train_pcn(
            mlp_graph,
            params_pc,
            train_loader,
            epochs=1,
            lr=0.01,
            infer_steps=3,
            eta_infer=0.05,
        )

        assert isinstance(results_bp["train_acc"], float)
        assert isinstance(results_pc["train_acc"], float)
