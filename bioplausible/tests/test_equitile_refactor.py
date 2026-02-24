import torch
import torch.nn as nn
import pytest
from bioplausible.models.equitile.task_handler import TaskHandler
from bioplausible.models.equitile.utils.init_utils import initialize_edge_weights, initialize_io_projections
from bioplausible.models.equitile.core import EquiTile
from bioplausible.models.equitile.enhanced import EnhancedEquiTile
from bioplausible.models.equitile.async_execution import AsyncEquiTile
from bioplausible.models.equitile.distributed import DistributedEquiTile

def test_task_handler_classification():
    handler = TaskHandler("classification", output_dim=3)
    logits = torch.randn(2, 3)
    y = torch.tensor([0, 2])

    loss = handler.compute_loss(logits, y)
    assert loss.dim() == 0
    assert loss.item() > 0

    loss, grad = handler.compute_loss_and_grad(logits, y)
    assert grad.shape == logits.shape

    acc = handler.compute_metrics(logits, y)
    assert 0 <= acc <= 1

def test_task_handler_regression():
    handler = TaskHandler("regression", output_dim=1)
    logits = torch.randn(2, 1)
    y = torch.randn(2)

    loss = handler.compute_loss(logits, y)
    assert loss.dim() == 0

    loss, grad = handler.compute_loss_and_grad(logits, y)
    assert grad.shape == logits.shape

    acc = handler.compute_metrics(logits, y)
    assert isinstance(acc, float)

def test_init_utils():
    weight = torch.empty(10, 20)
    initialize_edge_weights(weight, init_type="normal", gain=0.1)
    assert weight.std() < 1.0
    assert weight.std() > 0.0

    w_in = nn.Linear(5, 10)
    w_out = nn.Linear(10, 2)
    initialize_io_projections(w_in, w_out)
    assert w_in.weight.std() > 0
    assert w_out.weight.std() > 0

def test_equitile_training():
    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=5,
        output_dim=2
    )
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 0, 1])

    stats = model.train_step(x, y)
    assert "loss" in stats
    assert "accuracy" in stats

    logits = model(x)
    assert logits.shape == (4, 2)

def test_enhanced_equitile_training():
    model = EnhancedEquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=5,
        output_dim=2,
        use_layer_norm=True,
        use_batch_norm=True
    )
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 0, 1])

    stats = model.train_step(x, y)
    assert "loss" in stats
    assert "accuracy" in stats
    assert stats["enhanced"] is True

    logits = model(x)
    assert logits.shape == (4, 2)

def test_async_equitile_training():
    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=5,
        output_dim=2
    )
    # n_workers=1 to avoid issues in simple test env
    async_model = AsyncEquiTile(model, config=None)

    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 0, 1])

    # Test sync fallback (default)
    stats = async_model.train_step(x, y)
    assert "loss" in stats

    # Test async context (might be slow or tricky with threads in test env)
    # We just check it runs without error
    with async_model.async_context():
        stats = async_model.train_step(x, y)
        assert "loss" in stats

def test_distributed_equitile_training():
    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=5,
        output_dim=2
    )
    # Single device simulation (device_ids=[0] if cuda else cpu)
    dist_model = DistributedEquiTile(model)

    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 0, 1])

    stats = dist_model.train_step(x, y)
    assert "loss" in stats
    assert "accuracy" in stats
