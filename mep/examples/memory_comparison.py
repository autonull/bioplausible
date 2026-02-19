#!/usr/bin/env python3
"""
Memory Comparison: EP vs Backprop

This experiment measures peak memory usage for EP vs backprop
as network depth increases.

EP claims O(1) memory for activations vs O(depth) for backprop.
This script tests whether that translates to practical savings.

Run: python examples/memory_comparison.py
"""

import torch
import torch.nn as nn
import gc
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass

from mep import smep, muon_backprop


@dataclass
class MemoryResult:
    depth: int
    method: str
    peak_memory_mb: float
    success: bool
    error: Optional[str] = None
    train_time_sec: float = 0.0
    final_loss: float = 0.0


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e6


def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


class DeepMLP(nn.Module):
    """Deep MLP with configurable depth."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_with_backprop(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1,
    lr: float = 0.01
) -> Tuple[float, float]:
    """Train with backprop, return (peak_memory_mb, train_time)."""
    reset_memory()
    start = time.time()
    
    optimizer = muon_backprop(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    final_loss = 0.0
    
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        final_loss = loss.item()
        loss.backward()
        optimizer.step()
    
    train_time = time.time() - start
    peak_memory = get_gpu_memory_mb()
    
    return peak_memory, train_time, final_loss


def train_with_ep(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1,
    lr: float = 0.01
) -> Tuple[float, float]:
    """Train with EP, return (peak_memory_mb, train_time)."""
    reset_memory()
    start = time.time()
    
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=lr,
        mode='ep',
        settle_steps=10,
        settle_lr=0.1,
        loss_type='cross_entropy',
    )
    
    model.train()
    final_loss = 0.0
    
    for _ in range(epochs):
        optimizer.step(x=x, target=y)
        # Compute loss for reporting
        with torch.no_grad():
            output = model(x)
            final_loss = nn.functional.cross_entropy(output, y).item()
    
    train_time = time.time() - start
    peak_memory = get_gpu_memory_mb()
    
    return peak_memory, train_time, final_loss


def run_depth_experiment(
    depth: int,
    input_dim: int = 64,
    hidden_dim: int = 128,
    output_dim: int = 10,
    batch_size: int = 32,
    epochs: int = 1,
) -> Tuple[MemoryResult, MemoryResult]:
    """Run experiment for a single depth."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create data
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)
    
    bp_result = MemoryResult(depth=depth, method='backprop', peak_memory_mb=0, success=False)
    ep_result = MemoryResult(depth=depth, method='ep', peak_memory_mb=0, success=False)
    
    # Try backprop
    try:
        reset_memory()
        model_bp = DeepMLP(input_dim, hidden_dim, depth, output_dim).to(device)
        peak_mem, train_time, loss = train_with_backprop(model_bp, x, y, epochs=epochs)
        bp_result.peak_memory_mb = peak_mem
        bp_result.success = True
        bp_result.train_time_sec = train_time
        bp_result.final_loss = loss
        del model_bp
    except RuntimeError as e:
        bp_result.error = str(e)
        if 'out of memory' in str(e).lower():
            bp_result.error = 'OOM'
    
    # Try EP
    try:
        reset_memory()
        model_ep = DeepMLP(input_dim, hidden_dim, depth, output_dim).to(device)
        peak_mem, train_time, loss = train_with_ep(model_ep, x, y, epochs=epochs)
        ep_result.peak_memory_mb = peak_mem
        ep_result.success = True
        ep_result.train_time_sec = train_time
        ep_result.final_loss = loss
        del model_ep
    except RuntimeError as e:
        ep_result.error = str(e)
        if 'out of memory' in str(e).lower():
            ep_result.error = 'OOM'
    
    reset_memory()
    return bp_result, ep_result


def main():
    print("=" * 70)
    print("Memory Comparison: EP vs Backpropagation")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Running on CPU.")
        print("Memory measurements will be less meaningful.")
        print("For accurate results, run on GPU.\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test depths
    depths = [5, 10, 20, 50, 100, 200, 500]
    
    results: List[Tuple[MemoryResult, MemoryResult]] = []
    
    print(f"\n{'Depth':<8} {'Method':<12} {'Memory (MB)':<15} {'Time (s)':<12} {'Status':<10}")
    print("-" * 70)
    
    for depth in depths:
        bp_result, ep_result = run_depth_experiment(
            depth=depth,
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            batch_size=32,
            epochs=1,
        )
        results.append((bp_result, ep_result))
        
        bp_status = f"{bp_result.peak_memory_mb:.1f} MB" if bp_result.success else bp_result.error
        ep_status = f"{ep_result.peak_memory_mb:.1f} MB" if ep_result.success else ep_result.error
        
        bp_time = f"{bp_result.train_time_sec:.3f}" if bp_result.success else "N/A"
        ep_time = f"{ep_result.train_time_sec:.3f}" if ep_result.success else "N/A"
        
        print(f"{depth:<8} {'Backprop':<12} {bp_status:<15} {bp_time:<12} {'✓' if bp_result.success else '✗':<10}")
        print(f"{depth:<8} {'EP':<12} {ep_status:<15} {ep_time:<12} {'✓' if ep_result.success else '✗':<10}")
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    # Find crossover point
    bp_failed_at = None
    ep_failed_at = None
    
    for bp_result, ep_result in results:
        if not bp_result.success and bp_failed_at is None:
            bp_failed_at = bp_result.depth
        if not ep_result.success and ep_failed_at is None:
            ep_failed_at = ep_result.depth
    
    print(f"\nBackprop failed at depth: {bp_failed_at if bp_failed_at else 'Did not fail'}")
    print(f"EP failed at depth: {ep_failed_at if ep_failed_at else 'Did not fail'}")
    
    # Memory savings at each depth
    print(f"\n{'Depth':<8} {'BP Memory':<12} {'EP Memory':<12} {'Savings':<10}")
    print("-" * 50)
    for bp_result, ep_result in results:
        if bp_result.success and ep_result.success:
            savings = (1 - ep_result.peak_memory_mb / bp_result.peak_memory_mb) * 100
            print(f"{bp_result.depth:<8} {bp_result.peak_memory_mb:<12.1f} {ep_result.peak_memory_mb:<12.1f} {savings:>6.1f}%")
    
    print("\n" + "=" * 70)
    print("Conclusion:")
    
    if bp_failed_at and not ep_failed_at:
        print(f"  ✅ EP succeeded where backprop failed (depth {bp_failed_at})")
        print("  EP's O(1) memory advantage is REAL.")
    elif bp_failed_at and ep_failed_at and ep_failed_at > bp_failed_at:
        print(f"  ⚠️  EP failed at greater depth than backprop ({ep_failed_at} vs {bp_failed_at})")
        print("  EP has SOME memory advantage, but not unlimited.")
    elif not bp_failed_at and not ep_failed_at:
        # Compare memory at max depth
        bp_mem = results[-1][0].peak_memory_mb
        ep_mem = results[-1][1].peak_memory_mb
        savings = (1 - ep_mem / bp_mem) * 100
        if savings > 30:
            print(f"  ✅ EP uses {savings:.1f}% less memory at max depth")
            print("  EP's memory advantage is SIGNIFICANT.")
        elif savings > 10:
            print(f"  ⚠️  EP uses {savings:.1f}% less memory at max depth")
            print("  EP has MARGINAL memory advantage.")
        else:
            print(f"  ❌ EP uses {savings:.1f}% memory (no significant savings)")
            print("  EP's O(1) claim may not translate to practice.")
    else:
        print("  ❌ EP failed before backprop - no advantage demonstrated")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
