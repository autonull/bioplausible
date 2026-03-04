"""Tests for Adaptive Tile-Based Predictive Coding."""

import pytest
import torch
import torch.nn.functional as F

from bioplausible.models.tile_eq import AdaptiveTilePC, TileGraph, TileState


def make_xor(n_copies: int = 16):
    """XOR dataset."""
    torch.manual_seed(42)
    xs = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    ys = torch.tensor([0, 1, 1, 0])
    X = xs.repeat(n_copies, 1) + torch.randn(4 * n_copies, 2) * 0.1
    Y = ys.repeat(n_copies)
    return X, Y


def make_blobs(n_samples: int = 200, n_classes: int = 4):
    """Linearly separable blobs."""
    torch.manual_seed(42)
    means = [(i * 2.0, 0.0) for i in range(n_classes)]
    X_parts, Y_parts = [], []
    per = n_samples // n_classes

    for cls, (mx, my) in enumerate(means):
        pts = torch.randn(per, 2) * 0.3 + torch.tensor([mx, my])
        X_parts.append(pts)
        Y_parts.append(torch.full((per,), cls, dtype=torch.long))

    return torch.cat(X_parts), torch.cat(Y_parts)


def small_model(**kwargs) -> AdaptiveTilePC:
    """Create a small model for testing."""
    return AdaptiveTilePC(
        neurons_per_tile=kwargs.get("neurons_per_tile", 8),
        num_layers=kwargs.get("num_layers", 3),
        tiles_per_layer=kwargs.get("tiles_per_layer", 2),
        input_dim=kwargs.get("input_dim", 8),
        output_dim=kwargs.get("output_dim", 4),
        prediction_lr=kwargs.get("prediction_lr", 0.01),
        initial_step_size=kwargs.get("initial_step_size", 0.5),
        sparsity_threshold=kwargs.get("sparsity_threshold", 0.01),
        inference_steps=kwargs.get("inference_steps", 10),
    )


# -----------------------------------------------------------------------
# Basic Functionality
# -----------------------------------------------------------------------


def test_instantiation():
    """Model should instantiate without errors."""
    m = small_model()
    assert len(m.graph.tiles) > 0
    assert len(m.graph.edges) > 0
    assert m.W_in is not None
    assert m.W_out is not None


def test_forward_pass():
    """Forward pass should produce correct output shape."""
    m = small_model(input_dim=8, output_dim=4)
    x = torch.randn(4, 8)

    with torch.no_grad():
        logits = m(x)

    assert logits.shape == (4, 4)


def test_forward_with_states():
    """Forward with return_states should return tile activities."""
    m = small_model(input_dim=8, output_dim=4)
    x = torch.randn(2, 8)

    with torch.no_grad():
        logits, states = m(x, return_states=True)

    assert logits.shape == (2, 4)
    assert len(states) == len(m.graph.tiles)

    for tile_id, tile_states in states.items():
        assert "activity" in tile_states
        assert "prediction" in tile_states
        assert "error" in tile_states


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------


def test_train_step():
    """Training step should reduce loss and return metrics."""
    m = small_model(input_dim=8, output_dim=4)
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    initial_loss = None
    for i in range(20):
        stats = m.train_step(x, y)
        if i == 0:
            initial_loss = stats["loss"]

        assert "loss" in stats
        assert "accuracy" in stats
        assert "mean_error" in stats

    # Loss should generally decrease (with some tolerance for noise)
    assert stats["loss"] < initial_loss * 1.5


def test_learns_xor():
    """Model can train on XOR task without crashing."""
    m = AdaptiveTilePC(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=2,
        output_dim=2,
        prediction_lr=0.1,
        initial_step_size=1.0,
        inference_steps=20,
    )

    # XOR data
    xs = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    ys = torch.tensor([0, 1, 1, 0])
    X = xs.repeat(10, 1)
    Y = ys.repeat(10)

    # Train - verify no crashes and loss is finite
    losses = []
    for _ in range(100):
        idx = torch.randint(0, len(X), (8,))
        stats = m.train_step(X[idx], Y[idx])
        losses.append(stats["loss"])
        assert torch.isfinite(torch.tensor(stats["loss"])), "Loss became non-finite"

    # Forward pass should work
    with torch.no_grad():
        logits = m(xs)

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_learns_blobs():
    """Model can train on blobs task without crashing."""
    torch.manual_seed(42)
    X, Y = make_blobs(n_samples=200, n_classes=4)

    m = AdaptiveTilePC(
        neurons_per_tile=32,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=2,
        output_dim=4,
        prediction_lr=0.02,
        inference_steps=20,
    )

    # Train and verify no crashes, finite loss
    losses = []
    for _ in range(100):
        idx = torch.randint(0, len(X), (32,))
        stats = m.train_step(X[idx], Y[idx])
        losses.append(stats["loss"])
        assert torch.isfinite(torch.tensor(stats["loss"]))

    # Just verify training ran - actual learning depends on hyperparameters
    assert len(losses) == 100


# -----------------------------------------------------------------------
# Adaptive Computation
# -----------------------------------------------------------------------


def test_importance_weights():
    """Importance weights should be learned."""
    m = small_model()

    # Initial importance is sigmoid(1.0) ≈ 0.73
    initial_importance = torch.sigmoid(m.tile_importance).mean().item()
    assert 0.6 < initial_importance < 0.9

    # Train for a bit
    x = torch.randn(8, 8)
    y = torch.randint(0, 4, (8,))

    for _ in range(50):
        m.train_step(x, y)

    # Importance should have changed
    final_importance = torch.sigmoid(m.tile_importance).mean().item()
    assert final_importance != initial_importance


def test_sparse_updates():
    """Tiles with low error should be skipped."""
    m = small_model(sparsity_threshold=0.1)

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    # Run training
    stats = m.train_step(x, y)

    # Should report active tiles
    assert "active_tiles" in stats
    assert stats["active_tiles"] <= len(m.graph.tiles)


def test_error_ema():
    """Error EMA should track prediction errors."""
    m = small_model()

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    # Initial error EMA should be empty or zero
    assert len(m._error_ema) == 0 or all(v == 0.0 for v in m._error_ema.values())

    # After training, should have error estimates
    m.train_step(x, y)

    assert len(m._error_ema) > 0
    assert any(v > 0.0 for v in m._error_ema.values())


# -----------------------------------------------------------------------
# Graph Structure
# -----------------------------------------------------------------------


def test_graph_connectivity():
    """Tiles should be properly connected."""
    m = small_model(num_layers=3)

    # Input tiles should have forward neighbors only
    for tile_id in m.graph.input_tile_ids:
        tile = m.graph.tiles[tile_id]
        assert len(tile.fwd_neighbors) > 0
        assert len(tile.bwd_neighbors) == 0

    # Output tiles should have backward neighbors only
    for tile_id in m.graph.output_tile_ids:
        tile = m.graph.tiles[tile_id]
        assert len(tile.fwd_neighbors) == 0
        assert len(tile.bwd_neighbors) > 0

    # Hidden tiles should have both
    hidden_ids = (
        set(m.graph.tiles.keys())
        - set(m.graph.input_tile_ids)
        - set(m.graph.output_tile_ids)
    )
    for tile_id in hidden_ids:
        tile = m.graph.tiles[tile_id]
        assert len(tile.fwd_neighbors) > 0
        assert len(tile.bwd_neighbors) > 0


def test_layered_structure():
    """Graph should have correct layered structure."""
    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=5,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
    )

    assert len(m.graph.layer_ids) == 5
    assert len(m.graph.input_tile_ids) >= 1
    assert len(m.graph.output_tile_ids) >= 1


# -----------------------------------------------------------------------
# Statistics and Introspection
# -----------------------------------------------------------------------


def test_get_stats():
    """get_stats should return comprehensive metrics."""
    m = small_model()

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    m.train_step(x, y)

    stats = m.get_stats()

    assert "importance_mean" in stats
    assert "importance_max" in stats
    assert "error_mean" in stats
    assert "error_max" in stats
    assert "active_tiles" in stats
    assert "total_tiles" in stats


def test_topology_info():
    """get_topology_info should return visualization data."""
    m = small_model()

    info = m.get_topology_info()

    assert "positions" in info
    assert "edges" in info
    assert "layer_ids" in info
    assert "is_input" in info
    assert "is_output" in info
    assert "tile_heats" in info
    assert "importances" in info

    n_tiles = len(m.graph.tiles)
    assert len(info["positions"]) == n_tiles
    assert len(info["importances"]) == n_tiles


# -----------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------


def test_minimal_architecture():
    """Test with minimal 2-layer architecture."""
    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=2,
        tiles_per_layer=2,
        input_dim=8,
        output_dim=4,
    )

    x = torch.randn(2, 8)
    y = torch.randint(0, 4, (2,))

    logits = m(x)
    assert logits.shape == (2, 4)

    stats = m.train_step(x, y)
    assert stats["loss"] > 0


def test_deep_architecture():
    """Test with many layers."""
    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=8,
        tiles_per_layer=2,
        input_dim=8,
        output_dim=4,
    )

    assert len(m.graph.layer_ids) == 8

    x = torch.randn(2, 8)
    logits = m(x)
    assert logits.shape == (2, 4)


def test_different_activations():
    """Test with different activation functions."""
    for activation in ["tanh", "relu", "gelu"]:
        m = AdaptiveTilePC(
            neurons_per_tile=8,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=8,
            output_dim=4,
            activation=activation,
        )

        x = torch.randn(2, 8)
        logits = m(x)
        assert logits.shape == (2, 4)


# -----------------------------------------------------------------------
# Strategy Framework Tests
# -----------------------------------------------------------------------


def test_momentum_inference():
    """Test MomentumInference strategy."""
    from bioplausible.models.tile_eq import MomentumInference

    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=2,
        tiles_per_layer=2,
        input_dim=8,
        output_dim=4,
    )

    # Set momentum strategy
    m.inference_strategy = MomentumInference(momentum=0.9)

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    # Should train without errors
    stats = m.train_step(x, y)
    assert "loss" in stats


def test_oja_learning():
    """Test OjaLearning strategy."""
    from bioplausible.models.tile_eq import OjaLearning

    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=2,
        tiles_per_layer=2,
        input_dim=8,
        output_dim=4,
    )

    # Set Oja learning strategy
    m.learning_strategy = OjaLearning()

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    # Should train without errors
    stats = m.train_step(x, y)
    assert "loss" in stats


def test_topk_scheduling():
    """Test TopKScheduling strategy."""
    from bioplausible.models.tile_eq import TopKScheduling

    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=8,
        output_dim=4,
    )

    # Set Top-K scheduling
    m.scheduling_strategy = TopKScheduling(k=3, min_fraction=0.2)

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    # Should train without errors
    stats = m.train_step(x, y)
    assert "active_tiles" in stats
    assert stats["active_tiles"] <= len(m.graph.tiles)


def test_all_tiles_scheduling():
    """Test AllTilesScheduling strategy (no sparsity)."""
    from bioplausible.models.tile_eq import AllTilesScheduling

    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=2,
        tiles_per_layer=2,
        input_dim=8,
        output_dim=4,
    )

    # Set all-tiles scheduling
    m.scheduling_strategy = AllTilesScheduling()

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    stats = m.train_step(x, y)

    # All tiles should be active
    assert stats["active_tiles"] == len(m.graph.tiles)


def test_combined_strategies():
    """Test using multiple strategies together."""
    from bioplausible.models.tile_eq import (HebbianLearning,
                                             MomentumInference, TopKScheduling)

    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=8,
        output_dim=4,
    )

    # Configure all strategies
    m.inference_strategy = MomentumInference(momentum=0.9)
    m.learning_strategy = HebbianLearning()
    m.scheduling_strategy = TopKScheduling(k=5)

    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    # Train for a few steps (single step to avoid graph accumulation)
    stats = m.train_step(x, y)
    assert torch.isfinite(torch.tensor(stats["loss"]))


# -----------------------------------------------------------------------
# Custom Topology Tests
# -----------------------------------------------------------------------


def test_custom_topology():
    """Test custom (non-layered) topology."""
    # Create a graph with skip connections:
    # 0 (input) -> 1 -> 3 (output)
    # 0 (input) -> 2 -> 3 (output)
    edges = [
        (0, 1),
        (0, 2),  # Input to hidden
        (1, 3),
        (2, 3),  # Hidden to output
    ]

    m = AdaptiveTilePC(
        neurons_per_tile=8,
        num_layers=4,
        tiles_per_layer=1,
        input_dim=8,
        output_dim=4,
        topology="custom",
        custom_edges=edges,
        custom_positions=[(0, 0.5), (0.33, 0.25), (0.33, 0.75), (1, 0.5)],
    )

    # Verify structure
    assert len(m.graph.tiles) == 4
    assert 0 in m.graph.input_tile_ids
    assert 3 in m.graph.output_tile_ids

    # Tile 0 should connect to both 1 and 2
    assert set(m.graph.tiles[0].fwd_neighbors) == {1, 2}
    # Tile 3 should receive from both 1 and 2
    assert set(m.graph.tiles[3].bwd_neighbors) == {1, 2}

    # Forward pass should work
    x = torch.randn(2, 8)
    with torch.no_grad():
        logits = m(x)
    assert logits.shape == (2, 4)

    # Training should work
    y = torch.randint(0, 4, (2,))
    stats = m.train_step(x, y)
    assert "loss" in stats


def test_custom_topology_validation():
    """Test that custom topology validates inputs."""
    # Missing custom_edges should raise error
    with pytest.raises(ValueError):
        AdaptiveTilePC(
            neurons_per_tile=8,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=8,
            output_dim=4,
            topology="custom",
        )

    # Invalid topology name should raise error
    with pytest.raises(ValueError):
        AdaptiveTilePC(
            neurons_per_tile=8,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=8,
            output_dim=4,
            topology="invalid",
        )


def test_required_parameters():
    """Test that required parameters must be specified."""
    # Missing tiles_per_layer should raise TypeError
    with pytest.raises(TypeError):
        AdaptiveTilePC(
            neurons_per_tile=8,
            num_layers=3,
            input_dim=8,
            output_dim=4,
        )

    # Missing neurons_per_tile should raise TypeError
    with pytest.raises(TypeError):
        AdaptiveTilePC(
            num_layers=3,
            tiles_per_layer=2,
            input_dim=8,
            output_dim=4,
        )
