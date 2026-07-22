#!/usr/bin/env python3
"""
EquiTile Advanced Usage Examples

Demonstrates:
- Async tile execution
- Performance profiling
- Learning monitoring
- Multi-GPU concept

Usage:
    python examples/equitile_advanced_usage.py
"""

import torch

from bioplausible.models import (
    AsyncConfig,
    AsyncEquiTile,
    EquiTile,
    EquiTileProfiler,
    LearningMonitor,
)


def example_basic_profiling():
    """Example: Basic profiling."""
    print("=" * 70)
    print("Example 1: Basic Profiling")
    print("=" * 70)
    print()

    # Create model
    model = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
    )

    # Create profiler
    EquiTileProfiler(model)

    # Create data
    X = torch.randn(100, 32)
    y = torch.randint(0, 4, (100,))

    # Profile training step
    # Note: For full profiling, integrate profiler into model.train_step
    print("Profiling training step...")
    import time

    start = time.perf_counter()
    stats = model.train_step(X[:32], y[:32])
    elapsed = time.perf_counter() - start

    print()
    print(f"Training stats: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")
    print(f"Execution time: {elapsed*1000:.2f} ms")
    print()
    print("Note: Full tile-level profiling requires instrumenting train_step().")
    print("      Use LearningMonitor for easier training diagnostics.")
    print()


def example_learning_monitor():
    """Example: Learning monitoring."""
    print("=" * 70)
    print("Example 2: Learning Monitor")
    print("=" * 70)
    print()

    # Create model
    model = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
    )

    # Create monitor
    monitor = LearningMonitor(model, window_size=50)

    # Create data
    X = torch.randn(500, 32)
    y = torch.randint(0, 4, (500,))

    # Train with monitoring
    print("Training with monitoring...")
    for epoch in range(10):
        stats = model.train_step(X[:32], y[:32])
        monitor.record(stats)

        if epoch % 3 == 0:
            monitor.print_status()

    # Final summary
    summary = monitor.get_summary()
    print("Final Summary:")
    print(f"  Loss (avg): {summary['loss_mean']:.4f}")
    print(f"  Loss trend: {summary['loss_trend']}")
    print(f"  Accuracy (avg): {summary['accuracy_mean']:.4f}")
    print(f"  Accuracy trend: {summary['accuracy_trend']}")
    print()


def example_async_execution():
    """Example: Async tile execution."""
    print("=" * 70)
    print("Example 3: Async Tile Execution")
    print("=" * 70)
    print()

    # Create model
    model = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
    )

    # Create async wrapper
    async_model = AsyncEquiTile(
        model,
        config=AsyncConfig(
            n_workers=4,
            use_processes=False,  # Use threads (safer for demo)
            priority_alpha=0.5,
            priority_beta=0.5,
        ),
    )

    # Create data
    X = torch.randn(200, 32)
    y = torch.randint(0, 4, (200,))

    # Train with async context
    print("Training with async execution...")
    with async_model.async_context():
        for epoch in range(5):
            stats = async_model.train_step(X[:64], y[:64])
            print(
                f"  Epoch {epoch+1}: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}"
            )

    print()
    print("Note: Async execution shows benefits with larger models and batches.")
    print("      For small models, sync execution may be faster due to overhead.")
    print()


def example_tile_analysis():
    """Example: Tile-level analysis."""
    print("=" * 70)
    print("Example 4: Tile-Level Analysis")
    print("=" * 70)
    print()

    # Create model
    model = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=4,
        input_dim=32,
        output_dim=4,
    )

    # Create data
    X = torch.randn(100, 32)
    y = torch.randint(0, 4, (100,))

    # Train one step
    model.train_step(X[:32], y[:32])

    # Analyze tiles
    print("Tile Analysis:")
    print()

    for tile in model.graph.all_tiles:
        tile_idx = list(model.graph.tiles.keys()).index(tile.id)
        importance = torch.sigmoid(model.tile_importance[tile_idx]).item()

        activity_info = ""
        if tile.activity is not None:
            activity_info = (
                f"activity={tile.activity.mean().item():.4f}, "
                f"error={tile.error.norm().item():.4f}"
            )

        layer_type = (
            "input" if tile.is_input else ("output" if tile.is_output else "hidden")
        )

        print(
            f"  Tile {tile.id:2d} (layer {tile.layer_id}, {layer_type:6s}): "
            f"importance={importance:.3f}, {activity_info}"
        )

    print()

    # Find hot tiles
    errors = []
    for tile in model.graph.all_tiles:
        if tile.error is not None and not tile.is_input:
            errors.append((tile.id, tile.error.norm().item()))

    errors.sort(key=lambda x: x[1], reverse=True)

    print("Hot Tiles (highest error):")
    for tile_id, error_norm in errors[:3]:
        print(f"  Tile {tile_id}: error={error_norm:.4f}")

    print()


def example_multi_gpu_concept():
    """Example: Multi-GPU concept (demonstration only)."""
    print("=" * 70)
    print("Example 5: Multi-GPU Concept (Demonstration)")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("CUDA not available. Showing conceptual example.")
        print()
        print("Multi-GPU setup would distribute tiles across GPUs:")
        print("  GPU 0: Tiles 0-7")
        print("  GPU 1: Tiles 8-15")
        print("  ...")
        print()
        print("Each GPU processes its tiles independently.")
        print("Tile boundaries communicate via PCIe/NVLink.")
        return

    # Conceptual example
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs")
    print()

    # Create model
    model = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=4,
        input_dim=32,
        output_dim=4,
    )

    # Conceptual tile distribution
    n_tiles = len(model.graph.tiles)
    tiles_per_gpu = (n_tiles + n_gpus - 1) // n_gpus

    print(f"Tile distribution ({n_tiles} tiles across {n_gpus} GPUs):")
    for gpu_id in range(n_gpus):
        start_tile = gpu_id * tiles_per_gpu
        end_tile = min((gpu_id + 1) * tiles_per_gpu, n_tiles)
        if start_tile < n_tiles:
            print(f"  GPU {gpu_id}: Tiles {start_tile}-{end_tile-1}")

    print()
    print("Note: Full multi-GPU implementation requires:")
    print("  - Inter-GPU communication for tile boundaries")
    print("  - Gradient accumulation across devices")
    print("  - Synchronized weight updates (optional)")
    print()


def run_all_examples():
    """Run all examples."""
    print()
    print("=" * 70)
    print("EquiTile Advanced Usage Examples")
    print("=" * 70)
    print()

    example_basic_profiling()
    example_learning_monitor()
    example_async_execution()
    example_tile_analysis()
    example_multi_gpu_concept()

    print("=" * 70)
    print("Examples Complete")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Use EquiTileProfiler for performance debugging")
    print("  2. Use LearningMonitor for training progress")
    print("  3. Use AsyncEquiTile for parallel tile execution")
    print("  4. Explore multi-GPU setup for large models")
    print()


if __name__ == "__main__":
    run_all_examples()
