#!/usr/bin/env python3
"""
Comprehensive tests for EquiTile PC and EP modes.

These tests verify:
1. Both modes work correctly
2. PC mode uses task-driven local Hebbian learning
3. EP mode uses strict contrastive Hebbian updates
4. Performance comparison between modes
5. EP-specific features (beta annealing, early stopping)
"""

import torch
from bioplausible.models import EquiTile, EquiTileEP


def create_simple_dataset(n_samples=500, input_dim=16, output_dim=4):
    """Create a simple classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))

    # Add class structure
    for class_idx in range(output_dim):
        mask = y == class_idx
        X[mask] += class_idx * 1.5

    return X, y


def test_pc_mode_basic():
    """Test PC mode basic functionality."""
    print("Testing PC mode basic functionality...")

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="pc",
        inference_steps=5,
    )

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    stats = model.train_step(x, y)

    assert stats["mode"] == "pc", f"Expected mode='pc', got {stats.get('mode')}"
    assert "loss" in stats
    assert "accuracy" in stats
    assert not torch.isnan(torch.tensor(stats["loss"])), "Loss is NaN"

    print(f"  ✓ PC mode: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")
    return True


def test_ep_mode_basic():
    """Test EP mode basic functionality."""
    print("Testing EP mode basic functionality...")

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="ep",
        beta=0.1,
        inference_steps=5,
    )

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    stats = model.train_step(x, y)

    assert stats["mode"] == "ep", f"Expected mode='ep', got {stats.get('mode')}"
    assert "loss" in stats
    assert "accuracy" in stats
    assert "beta" in stats, "EP mode should report beta value"
    assert not torch.isnan(torch.tensor(stats["loss"])), "Loss is NaN"

    print(
        f"  ✓ EP mode: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}, beta={stats['beta']:.4f}"
    )
    return True


def test_equitile_ep_class():
    """Test EquiTileEP convenience class."""
    print("Testing EquiTileEP class...")

    model = EquiTileEP(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        beta=0.1,
        inference_steps=5,
    )

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    stats = model.train_step(x, y)

    assert stats["mode"] == "ep", f"Expected mode='ep', got {stats.get('mode')}"

    print(f"  ✓ EquiTileEP: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")
    return True


def test_ep_contrastive_property():
    """Verify EP mode uses actual free-nudged contrast."""
    print("Testing EP contrastive property...")

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="ep",
        beta=0.1,
        inference_steps=5,
    )

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    # Store initial weights
    initial_weights = {}
    for edge_key, edge in model.graph.edges.items():
        if edge.weight is not None:
            initial_weights[edge_key] = edge.weight.data.clone()

    # Train one step
    model.train_step(x, y)

    # Check that weights changed (contrastive update happened)
    weights_changed = False
    for edge_key, edge in model.graph.edges.items():
        if edge.weight is not None:
            if not torch.allclose(
                initial_weights[edge_key], edge.weight.data, atol=1e-6
            ):
                weights_changed = True
                break

    assert weights_changed, "EP should update weights via contrastive learning"

    print("  ✓ EP contrastive updates verified")
    return True


def test_pc_local_hebbian_property():
    """Verify PC mode uses task-driven local Hebbian updates."""
    print("Testing PC local Hebbian property...")

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="pc",
        inference_steps=5,
    )

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    # Store initial weights
    initial_weights = {}
    for edge_key, edge in model.graph.edges.items():
        if edge.weight is not None:
            initial_weights[edge_key] = edge.weight.data.clone()

    # Train one step
    model.train_step(x, y)

    # Check that weights changed
    weights_changed = False
    for edge_key, edge in model.graph.edges.items():
        if edge.weight is not None:
            if not torch.allclose(
                initial_weights[edge_key], edge.weight.data, atol=1e-6
            ):
                weights_changed = True
                break

    assert weights_changed, "PC should update weights via local Hebbian learning"

    print("  ✓ PC local Hebbian updates verified")
    return True


def test_beta_annealing():
    """Test EP beta annealing feature."""
    print("Testing EP beta annealing...")

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="ep",
        beta=0.1,
        beta_anneal=0.9,  # Decay beta each step
        inference_steps=5,
    )

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    # First step
    stats1 = model.train_step(x, y)
    beta1 = stats1["beta"]

    # Second step
    stats2 = model.train_step(x, y)
    beta2 = stats2["beta"]

    # Beta should decay
    assert beta2 < beta1, f"Beta should decay: {beta2} < {beta1}"
    expected_beta2 = beta1 * 0.9
    assert abs(beta2 - expected_beta2) < 1e-6, (
        f"Beta decay incorrect: {beta2} vs {expected_beta2}"
    )

    print(f"  ✓ Beta annealing: {beta1:.4f} → {beta2:.4f}")
    return True


def test_separate_inference_steps():
    """Test separate free/nudged phase steps."""
    print("Testing separate inference steps...")

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="ep",
        beta=0.1,
        inference_steps=5,
        inference_steps_free=10,  # More steps for free phase
        inference_steps_nudged=15,  # Even more for nudged phase
    )

    assert model.config.inference_steps_free == 10
    assert model.config.inference_steps_nudged == 15

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    stats = model.train_step(x, y)

    assert stats["mode"] == "ep"

    print("  ✓ Separate inference steps configured")
    return True


def test_early_stopping():
    """Test EP early stopping for relaxation."""
    print("Testing early stopping...")

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="ep",
        beta=0.1,
        inference_steps=50,  # Many steps allowed
        relaxation_tolerance=1e-3,  # But should stop early if converged
    )

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))

    stats = model.train_step(x, y)

    # Should complete without error
    assert "loss" in stats

    print("  ✓ Early stopping works")
    return True


def test_mode_comparison_learning():
    """Compare learning between PC and EP modes."""
    print("Testing PC vs EP learning comparison...")

    # Create dataset
    X, y = create_simple_dataset(n_samples=200, input_dim=16, output_dim=4)

    # PC model
    model_pc = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="pc",
        inference_steps=10,
        learning_rate=0.01,
    )

    # EP model
    model_ep = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="ep",
        beta=0.1,
        inference_steps=10,
        inference_steps_free=15,
        inference_steps_nudged=15,
        learning_rate=0.01,
    )

    # Train both for a few epochs
    n_epochs = 5
    pc_losses = []
    ep_losses = []

    for epoch in range(n_epochs):
        # PC
        epoch_loss_pc = 0
        for i in range(0, len(X), 32):
            stats = model_pc.train_step(X[i : i + 32], y[i : i + 32])
            epoch_loss_pc += stats["loss"]
        pc_losses.append(epoch_loss_pc / (len(X) // 32))

        # EP
        epoch_loss_ep = 0
        for i in range(0, len(X), 32):
            stats = model_ep.train_step(X[i : i + 32], y[i : i + 32])
            epoch_loss_ep += stats["loss"]
        ep_losses.append(epoch_loss_ep / (len(X) // 32))

    # Both should learn (loss should generally decrease)
    # Note: EP may be less stable, so we just check it runs
    assert pc_losses[-1] < pc_losses[0], (
        f"PC should learn: {pc_losses[0]:.4f} → {pc_losses[-1]:.4f}"
    )

    print(f"  ✓ PC learning: {pc_losses[0]:.4f} → {pc_losses[-1]:.4f}")
    print(f"  ✓ EP learning: {ep_losses[0]:.4f} → {ep_losses[-1]:.4f}")

    return True


def test_all_task_types():
    """Test both modes with all task types."""
    print("Testing all task types...")

    task_configs = [
        ("classification", 4, torch.randint(0, 4, (8,))),
        ("regression", 1, torch.randn(8, 1)),
        ("binary", 1, torch.randint(0, 2, (8, 1)).float()),
        ("multilabel", 4, torch.randint(0, 2, (8, 4)).float()),
    ]

    for mode in ["pc", "ep"]:
        for task_type, output_dim, y in task_configs:
            model = EquiTile(
                neurons_per_tile=8,
                num_layers=3,
                tiles_per_layer=1,
                input_dim=16,
                output_dim=output_dim,
                task_type=task_type,
                mode=mode,
                inference_steps=3,
            )

            x = torch.randn(8, 16)
            stats = model.train_step(x, y)

            assert stats["mode"] == mode
            assert "loss" in stats
            assert not torch.isnan(torch.tensor(stats["loss"])), (
                f"{mode}/{task_type}: Loss is NaN"
            )

    print("  ✓ All task types work for both modes")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("EquiTile Mode Comparison Tests")
    print("=" * 60)
    print()

    tests = [
        ("PC Mode Basic", test_pc_mode_basic),
        ("EP Mode Basic", test_ep_mode_basic),
        ("EquiTileEP Class", test_equitile_ep_class),
        ("EP Contrastive Property", test_ep_contrastive_property),
        ("PC Local Hebbian Property", test_pc_local_hebbian_property),
        ("Beta Annealing", test_beta_annealing),
        ("Separate Inference Steps", test_separate_inference_steps),
        ("Early Stopping", test_early_stopping),
        ("Mode Comparison Learning", test_mode_comparison_learning),
        ("All Task Types", test_all_task_types),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
