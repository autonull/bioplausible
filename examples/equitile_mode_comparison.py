#!/usr/bin/env python3
"""
EquiTile Mode Comparison Example

This script demonstrates the differences between PC mode (predictive coding)
and EP mode (strict Equilibrium Propagation) on a simple classification task.

Usage:
    python examples/equitile_mode_comparison.py
"""

import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from bioplausible.equitile import EquiTile


def create_dataset(n_samples=1000, input_dim=32, output_dim=4):
    """Create a simple classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))

    # Add class structure
    for class_idx in range(output_dim):
        mask = y == class_idx
        X[mask] += class_idx * 1.5

    return X, y


def train_model(model, X, y, n_epochs=10, batch_size=32, verbose=True):
    """Train a model and return history."""
    history = {"loss": [], "accuracy": [], "time_per_epoch": []}

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        start_time = time.time()

        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0

        for batch_x, batch_y in loader:
            stats = model.train_step(batch_x, batch_y)
            epoch_loss += stats["loss"]
            epoch_acc += stats["accuracy"]
            n_batches += 1

        epoch_time = time.time() - start_time

        history["loss"].append(epoch_loss / n_batches)
        history["accuracy"].append(epoch_acc / n_batches)
        history["time_per_epoch"].append(epoch_time)

        if verbose:
            stats.get("mode", "N/A")
            beta_str = f", β={stats.get('beta', 0):.3f}" if "beta" in stats else ""
            print(
                f"  Epoch {epoch+1:3d}: Loss={epoch_loss/n_batches:.4f}, "
                f"Acc={epoch_acc/n_batches:.4f}{beta_str}, "
                f"Time={epoch_time:.2f}s"
            )

    return history


def compare_modes():
    """Compare PC and EP modes side by side."""
    print("=" * 70)
    print("EquiTile Mode Comparison: PC vs EP")
    print("=" * 70)
    print()

    # Create dataset
    print("Creating dataset...")
    X_train, y_train = create_dataset(n_samples=500, input_dim=32, output_dim=4)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Input dim: {X_train.shape[1]}, Classes: {len(torch.unique(y_train))}")
    print()

    # PC Model
    print("Creating PC mode model (Predictive Coding + Local Hebbian)...")
    model_pc = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
        mode="pc",
        inference_steps=10,
        learning_rate=0.01,
        dropout=0.1,
    )
    print(f"  Parameters: {sum(p.numel() for p in model_pc.parameters()):,}")
    print()

    # EP Model
    print("Creating EP mode model (Strict Equilibrium Propagation)...")
    model_ep = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
        mode="ep",
        beta=0.1,
        beta_anneal=0.99,  # Slow beta decay
        inference_steps=10,
        inference_steps_free=15,
        inference_steps_nudged=15,
        learning_rate=0.01,
        dropout=0.1,
    )
    print(f"  Parameters: {sum(p.numel() for p in model_ep.parameters()):,}")
    print()

    # Train PC
    print("Training PC mode...")
    print("-" * 70)
    history_pc = train_model(model_pc, X_train, y_train, n_epochs=15, batch_size=32)
    print()

    # Train EP
    print("Training EP mode...")
    print("-" * 70)
    history_ep = train_model(model_ep, X_train, y_train, n_epochs=15, batch_size=32)
    print()

    # Compare results
    print("=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print()

    # Final performance
    final_acc_pc = history_pc["accuracy"][-1]
    final_acc_ep = history_ep["accuracy"][-1]

    print("Final Training Accuracy:")
    print(f"  PC mode: {final_acc_pc*100:.1f}%")
    print(f"  EP mode: {final_acc_ep*100:.1f}%")
    print()

    # Learning speed
    print("Average Time per Epoch:")
    print(
        f"  PC mode: {sum(history_pc['time_per_epoch'])/len(history_pc['time_per_epoch']):.2f}s"
    )
    print(
        f"  EP mode: {sum(history_ep['time_per_epoch'])/len(history_ep['time_per_epoch']):.2f}s"
    )
    print()

    # Loss improvement
    loss_improve_pc = history_pc["loss"][0] - history_pc["loss"][-1]
    loss_improve_ep = history_ep["loss"][0] - history_ep["loss"][-1]

    print("Loss Improvement:")
    print(
        f"  PC mode: {history_pc['loss'][0]:.4f} → {history_pc['loss'][-1]:.4f} (Δ={loss_improve_pc:.4f})"
    )
    print(
        f"  EP mode: {history_ep['loss'][0]:.4f} → {history_ep['loss'][-1]:.4f} (Δ={loss_improve_ep:.4f})"
    )
    print()

    # Summary
    print("=" * 70)
    print("Summary & Recommendations")
    print("=" * 70)
    print()

    if final_acc_pc > final_acc_ep:
        print("✓ PC mode achieved higher accuracy")
    else:
        print("✓ EP mode achieved higher accuracy")

    if history_pc["time_per_epoch"][0] < history_ep["time_per_epoch"][0]:
        print("✓ PC mode is faster per epoch")
    else:
        print("✓ EP mode is faster per epoch")

    print()
    print("When to use each mode:")
    print()
    print("PC Mode (Predictive Coding + Local Hebbian):")
    print("  ✓ Default choice for most applications")
    print("  ✓ Strong, stable learning")
    print("  ✓ Good bio-plausibility with practical performance")
    print("  ✓ Recommended for: classification, regression, standard tasks")
    print()
    print("EP Mode (Strict Equilibrium Propagation):")
    print("  ✓ Research applications requiring strict EP")
    print("  ✓ Maximum biological plausibility")
    print("  ✓ No error backpropagation through graph")
    print("  ✓ Recommended for: neuroscience research, EP algorithm studies")
    print()

    return history_pc, history_ep


def demo_beta_annealing():
    """Demonstrate beta annealing in EP mode."""
    print("=" * 70)
    print("EP Mode: Beta Annealing Demo")
    print("=" * 70)
    print()

    model = EquiTile(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        mode="ep",
        beta=0.2,
        beta_anneal=0.9,  # Decay by 10% each step
        inference_steps=5,
    )

    X, y = create_dataset(n_samples=100, input_dim=16, output_dim=4)

    print("Training with beta annealing (β starts at 0.2, decays by 0.9× each step):")
    print()

    for epoch in range(5):
        stats = model.train_step(X[:32], y[:32])
        print(f"  Step {epoch+1}: β={stats['beta']:.4f}, Loss={stats['loss']:.4f}")

    print()
    print("Beta annealing helps EP converge by:")
    print("  1. Starting with strong nudge signal (high β)")
    print("  2. Gradually reducing nudge as learning progresses")
    print("  3. Allowing finer adjustments near convergence")
    print()


if __name__ == "__main__":
    # Run comparison
    compare_modes()

    print()
    print()

    # Demo beta annealing
    demo_beta_annealing()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
