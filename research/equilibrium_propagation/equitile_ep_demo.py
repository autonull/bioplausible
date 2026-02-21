#!/usr/bin/env python3
"""
Equilibrium Propagation Research Demo

This script demonstrates strict Equilibrium Propagation (EP) mode for research purposes.

EP is archived as experimental because:
- Lower accuracy than PC mode (~23% vs ~97%)
- Slower training (2-3× more compute)
- Requires careful hyperparameter tuning

However, EP is theoretically interesting for:
- Strictly local learning (no error backpropagation)
- Bio-plausible credit assignment research
- Energy-based model connections

Usage:
    python research/equilibrium_propagation/equitile_ep_demo.py
"""

import torch
from bioplausible.models import EquiTile, EquiTileEP


def create_dataset(n_samples=200, input_dim=16, output_dim=4):
    """Create a simple classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    for class_idx in range(output_dim):
        mask = y == class_idx
        X[mask] += class_idx * 1.5
    return X, y


def demo_basic_ep():
    """Demonstrate basic EP mode usage."""
    print("=" * 60)
    print("Equilibrium Propagation Demo")
    print("=" * 60)
    print()
    
    # Create model with EP mode
    model = EquiTileEP(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        beta=0.1,
        beta_anneal=0.99,  # Slow beta decay
        inference_steps_free=15,
        inference_steps_nudged=15,
        learning_rate=0.01,
    )
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Mode: {model.mode}")
    print(f"Beta: {model.config.beta} (anneal: {model.config.beta_anneal})")
    print()
    
    # Create dataset
    X, y = create_dataset()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(torch.unique(y))} classes")
    print()
    
    # Train
    print("Training (EP mode)...")
    print("-" * 60)
    
    for epoch in range(10):
        stats = model.train_step(X[:32], y[:32])
        print(f"  Epoch {epoch+1:3d}: Loss={stats['loss']:.4f}, "
              f"Acc={stats['accuracy']:.4f}, β={stats['beta']:.4f}")
    
    print()
    print("Note: EP mode typically achieves lower accuracy than PC mode.")
    print("      This is expected due to the strict locality constraint.")
    print()


def demo_ep_parameters():
    """Demonstrate EP-specific parameters."""
    print("=" * 60)
    print("EP Mode Parameters")
    print("=" * 60)
    print()
    
    # Default EP
    model1 = EquiTile(mode='ep', neurons_per_tile=16, num_layers=3,
                      tiles_per_layer=2, input_dim=16, output_dim=4)
    print(f"Default EP: β={model1.config.beta}, "
          f"steps_free={model1.config.inference_steps_free}, "
          f"steps_nudged={model1.config.inference_steps_nudged}")
    
    # Tuned EP
    model2 = EquiTile(
        mode='ep',
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        beta=0.2,           # Higher initial beta
        beta_anneal=0.95,   # Faster decay
        inference_steps_free=25,   # More free phase steps
        inference_steps_nudged=25, # More nudged phase steps
        relaxation_tolerance=1e-5, # Tighter convergence
    )
    print(f"Tuned EP:   β={model2.config.beta}, "
          f"anneal={model2.config.beta_anneal}, "
          f"steps_free={model2.config.inference_steps_free}, "
          f"steps_nudged={model2.config.inference_steps_nudged}")
    
    print()
    print("Parameter tips:")
    print("  - Higher beta: Stronger nudge signal, may destabilize")
    print("  - Beta anneal < 1.0: Decay beta over time for fine-tuning")
    print("  - More inference steps: Better convergence, slower training")
    print("  - Lower tolerance: More precise equilibrium, more compute")
    print()


def demo_contrastive_property():
    """Demonstrate the contrastive learning property of EP."""
    print("=" * 60)
    print("EP Contrastive Learning Property")
    print("=" * 60)
    print()
    
    model = EquiTileEP(
        neurons_per_tile=16,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=16,
        output_dim=4,
        beta=0.1,
        inference_steps=5,
    )
    
    X, y = create_dataset()
    
    # Get initial weights
    edge_key = list(model.graph.edges.keys())[0]
    initial_weight = model.graph.edges[edge_key].weight.data.clone()
    
    # Train one step
    model.train_step(X[:8], y[:8])
    
    # Get updated weights
    updated_weight = model.graph.edges[edge_key].weight.data
    
    # Compute change
    weight_change = (updated_weight - initial_weight).abs().mean().item()
    
    print(f"Weight update analysis (single training step):")
    print(f"  Edge: {edge_key}")
    print(f"  Initial weight mean: {initial_weight.mean().item():.6f}")
    print(f"  Updated weight mean: {updated_weight.mean().item():.6f}")
    print(f"  Mean absolute change: {weight_change:.6f}")
    print()
    print("EP weight update rule:")
    print("  ΔW = (η/β) × (pre_free·post_free - pre_nudged·post_nudged)")
    print()
    print("This is purely local—no error backpropagation through the graph.")
    print()


if __name__ == "__main__":
    demo_basic_ep()
    demo_ep_parameters()
    demo_contrastive_property()
    
    print("=" * 60)
    print("Demo Complete")
    print("=" * 60)
    print()
    print("For production use, see EquiTile PC mode:")
    print("  from bioplausible.models import EquiTile")
    print("  model = EquiTile(mode='pc', ...)  # or just EquiTile(...)")
    print()
    print("For more EP research, see:")
    print("  - Scellier & Bengio (2017). Equilibrium Propagation.")
    print("  - Laborieux et al. (2021). Scaling EP to Deep ConvNets.")
    print()
