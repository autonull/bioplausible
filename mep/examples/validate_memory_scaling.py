#!/usr/bin/env python3
"""
Memory Scaling Validation: EP vs Backprop with Gradient Checkpointing

This script properly validates EP's O(1) activation memory claim by:
1. Using gradient checkpointing to isolate activation memory from weight memory
2. Testing at extreme depths (100, 500, 1000, 2000+ layers)
3. Measuring only activation memory, not total memory
4. Following bioplausible Track 35 methodology

Key insight: Backprop stores O(depth) activations for gradient computation.
EP only stores current states (O(1)) since it uses iterative settling.

Run: python examples/validate_memory_scaling.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, asdict
import json

from mep import smep, muon_backprop


@dataclass
class MemoryMeasurement:
    """Memory measurement for a single configuration."""
    depth: int
    method: str
    checkpointing: bool
    activation_memory_mb: float
    total_memory_mb: float
    weight_memory_mb: float
    success: bool
    error: Optional[str] = None
    train_time_sec: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0


def get_memory_stats() -> Dict[str, float]:
    """Get detailed GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1e6,
        'reserved_mb': torch.cuda.memory_reserved() / 1e6,
        'peak_allocated_mb': torch.cuda.memory_stats().get('allocated_bytes.all.peak', 0) / 1e6,
        'peak_reserved_mb': torch.cuda.memory_stats().get('reserved_bytes.all.peak', 0) / 1e6,
    }


def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


class DeepMLP(nn.Module):
    """Deep MLP with configurable depth for memory testing."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.num_layers = num_layers
        
        # Create layers as ModuleList for checkpointing compatibility
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
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CheckpointedDeepMLP(nn.Module):
    """Deep MLP with gradient checkpointing for fair memory comparison."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers with checkpointing
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply gradient checkpointing to each layer pair (Linear + ReLU)
        # This isolates activation memory from O(depth) storage
        x = self.network[0](x)  # First linear
        x = F.relu(x)
        
        for i in range(1, len(self.network) - 1, 2):
            linear = self.network[i]
            # Checkpoint the linear transformation
            x = torch.utils.checkpoint.checkpoint(linear, x, use_reentrant=False)
            x = F.relu(x)
        
        # Output layer
        x = self.network[-1](x)
        return x


def measure_weight_memory(model: nn.Module) -> float:
    """Measure memory used by model weights only."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assume 4 bytes per float32 parameter
    return total_params * 4 / 1e6


def train_step_backprop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[float, Dict[str, float]]:
    """Single training step with backprop, return loss and memory stats."""
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    return loss.item(), get_memory_stats()


def train_step_ep(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[float, Dict[str, float]]:
    """Single training step with EP, return loss and memory stats."""
    optimizer.step(x=x, target=y)
    
    # Compute loss for reporting
    with torch.no_grad():
        output = model(x)
        loss = F.cross_entropy(output, y).item()
    
    return loss, get_memory_stats()


def measure_activation_memory(
    model: nn.Module,
    method: str,
    x: torch.Tensor,
    y: torch.Tensor,
    use_checkpointing: bool = False,
    lr: float = 0.01,
) -> MemoryMeasurement:
    """
    Measure activation memory for a single forward-backward pass.
    
    Key: We measure the PEAK memory during training, then subtract
    weight memory to isolate activation memory.
    """
    reset_memory()
    
    # Get baseline memory (model weights only)
    weight_mem = measure_weight_memory(model)
    baseline_mem = get_memory_stats().get('allocated_mb', 0)
    
    # Create optimizer
    if method == 'backprop':
        optimizer = muon_backprop(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
    else:  # EP
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=lr,
            mode='ep',
            settle_steps=10,
            settle_lr=0.1,
            loss_type='cross_entropy',
        )
    
    # Training step and measure peak
    start = time.time()
    
    if method == 'backprop':
        _, mem_stats = train_step_backprop(model, optimizer, criterion, x, y)
    else:
        _, mem_stats = train_step_ep(model, optimizer, x, y)
    
    train_time = time.time() - start
    
    # Calculate activation memory
    peak_allocated = mem_stats.get('peak_allocated_mb', 0)
    activation_mem = peak_allocated - weight_mem
    
    return MemoryMeasurement(
        depth=model.num_layers,
        method=method,
        checkpointing=use_checkpointing,
        activation_memory_mb=max(0, activation_mem),
        total_memory_mb=peak_allocated,
        weight_memory_mb=weight_mem,
        success=True,
        train_time_sec=train_time,
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=mem_stats.get('peak_reserved_mb', 0),
    )


def run_scaling_experiment(
    depths: List[int],
    input_dim: int = 64,
    hidden_dim: int = 128,
    output_dim: int = 10,
    batch_size: int = 32,
    use_checkpointing: bool = True,
) -> List[MemoryMeasurement]:
    """Run memory scaling experiment for all depths."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Checkpointing: {use_checkpointing}")
    print(f"Hidden dim: {hidden_dim}, Batch size: {batch_size}")
    
    results = []
    
    # Create data
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)
    
    for depth in depths:
        print(f"\nTesting depth {depth}...")
        
        # Test backprop
        if use_checkpointing:
            model_bp = CheckpointedDeepMLP(input_dim, hidden_dim, depth, output_dim).to(device)
        else:
            model_bp = DeepMLP(input_dim, hidden_dim, depth, output_dim).to(device)
        
        try:
            result_bp = measure_activation_memory(
                model_bp, 'backprop', x, y, use_checkpointing
            )
            results.append(result_bp)
            print(f"  Backprop: {result_bp.activation_memory_mb:.2f} MB (activation), "
                  f"{result_bp.total_memory_mb:.2f} MB (total)")
            del model_bp
        except RuntimeError as e:
            results.append(MemoryMeasurement(
                depth=depth, method='backprop', checkpointing=use_checkpointing,
                activation_memory_mb=0, total_memory_mb=0, weight_memory_mb=0,
                success=False, error=str(e)
            ))
            print(f"  Backprop: FAILED - {e}")
        
        reset_memory()
        
        # Test EP
        model_ep = DeepMLP(input_dim, hidden_dim, depth, output_dim).to(device)
        
        try:
            result_ep = measure_activation_memory(
                model_ep, 'ep', x, y, use_checkpointing
            )
            results.append(result_ep)
            print(f"  EP:       {result_ep.activation_memory_mb:.2f} MB (activation), "
                  f"{result_ep.total_memory_mb:.2f} MB (total)")
            del model_ep
        except RuntimeError as e:
            results.append(MemoryMeasurement(
                depth=depth, method='ep', checkpointing=use_checkpointing,
                activation_memory_mb=0, total_memory_mb=0, weight_memory_mb=0,
                success=False, error=str(e)
            ))
            print(f"  EP:       FAILED - {e}")
        
        reset_memory()
    
    return results


def analyze_scaling(results: List[MemoryMeasurement]) -> Dict:
    """Analyze memory scaling behavior."""
    
    # Group by method
    bp_results = [r for r in results if r.method == 'backprop' and r.success]
    ep_results = [r for r in results if r.method == 'ep' and r.success]
    
    if not bp_results or not ep_results:
        return {'error': 'Insufficient data for analysis'}
    
    # Fit linear scaling for backprop
    bp_depths = [r.depth for r in bp_results]
    bp_mem = [r.activation_memory_mb for r in bp_results]
    
    ep_depths = [r.depth for r in ep_results]
    ep_mem = [r.activation_memory_mb for r in ep_results]
    
    # Calculate scaling ratios
    if len(bp_results) >= 2:
        bp_scaling = (bp_mem[-1] - bp_mem[0]) / (bp_depths[-1] - bp_depths[0])
    else:
        bp_scaling = 0
    
    if len(ep_results) >= 2:
        ep_scaling = (ep_mem[-1] - ep_mem[0]) / (ep_depths[-1] - ep_depths[0])
    else:
        ep_scaling = 0
    
    # Memory savings at max depth
    max_depth_bp = bp_results[-1]
    max_depth_ep = ep_results[-1]
    savings = (1 - max_depth_ep.activation_memory_mb / max_depth_bp.activation_memory_mb) * 100
    
    return {
        'bp_scaling_mb_per_layer': bp_scaling,
        'ep_scaling_mb_per_layer': ep_scaling,
        'memory_savings_at_max_depth_pct': savings,
        'bp_max_depth_mem_mb': max_depth_bp.activation_memory_mb,
        'ep_max_depth_mem_mb': max_depth_ep.activation_memory_mb,
        'crossover_depth': None,  # Would need more analysis
    }


def print_results_table(results: List[MemoryMeasurement]):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("MEMORY SCALING RESULTS")
    print("=" * 90)
    print(f"{'Depth':<8} {'Method':<12} {'Checkpoint':<12} {'Activation MB':<15} {'Total MB':<12} {'Time (s)':<10} {'Status':<8}")
    print("-" * 90)
    
    for r in results:
        checkpoint = "Yes" if r.checkpointing else "No"
        act_mem = f"{r.activation_memory_mb:.2f}" if r.success else "N/A"
        total_mem = f"{r.total_memory_mb:.2f}" if r.success else "N/A"
        time_s = f"{r.train_time_sec:.3f}" if r.success else "N/A"
        status = "✓" if r.success else "✗"
        
        print(f"{r.depth:<8} {r.method:<12} {checkpoint:<12} {act_mem:<15} {total_mem:<12} {time_s:<10} {status:<8}")
    
    print("=" * 90)


def plot_results(results: List[MemoryMeasurement], save_path: Optional[str] = None):
    """Plot memory scaling results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
        return
    
    # Filter successful results
    bp_results = [r for r in results if r.method == 'backprop' and r.success]
    ep_results = [r for r in results if r.method == 'ep' and r.success]
    
    if not bp_results or not ep_results:
        print("\nInsufficient data for plotting.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Activation memory vs depth
    ax1 = axes[0]
    ax1.plot([r.depth for r in bp_results], [r.activation_memory_mb for r in bp_results], 
             'o-', label='Backprop', linewidth=2, markersize=8)
    ax1.plot([r.depth for r in ep_results], [r.activation_memory_mb for r in ep_results], 
             's-', label='EP', linewidth=2, markersize=8)
    ax1.set_xlabel('Network Depth (layers)', fontsize=12)
    ax1.set_ylabel('Activation Memory (MB)', fontsize=12)
    ax1.set_title('Activation Memory Scaling with Depth', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory savings percentage
    ax2 = axes[1]
    depths = sorted(set(r.depth for r in results if r.success))
    savings = []
    for depth in depths:
        bp_mem = next((r.activation_memory_mb for r in bp_results if r.depth == depth), None)
        ep_mem = next((r.activation_memory_mb for r in ep_results if r.depth == depth), None)
        if bp_mem and ep_mem and bp_mem > 0:
            savings.append((1 - ep_mem / bp_mem) * 100)
        else:
            savings.append(None)
    
    ax2.bar(range(len(depths)), [s if s else 0 for s in savings], color='green', alpha=0.6)
    ax2.set_xlabel('Network Depth (layers)', fontsize=12)
    ax2.set_ylabel('Memory Savings (%)', fontsize=12)
    ax2.set_title('EP Memory Savings vs Backprop', fontsize=14)
    ax2.set_xticks(range(len(depths)))
    ax2.set_xticklabels([str(d) for d in depths])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def save_results(results: List[MemoryMeasurement], save_path: str):
    """Save results to JSON file."""
    data = [asdict(r) for r in results]
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {save_path}")


def main():
    print("=" * 90)
    print("MEMORY SCALING VALIDATION: EP vs Backprop")
    print("Proper validation using gradient checkpointing methodology")
    print("=" * 90)
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Running on CPU.")
        print("Memory measurements will be less meaningful.")
        print("For accurate results, run on GPU.\n")
    
    # Test depths - focus on extreme depths where scaling matters
    depths = [10, 50, 100, 200, 500, 1000, 2000]
    
    # Run with gradient checkpointing (fair comparison)
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: With Gradient Checkpointing (Fair Comparison)")
    print("=" * 90)
    
    results_checkpoint = run_scaling_experiment(
        depths=depths,
        use_checkpointing=True,
    )
    
    print_results_table(results_checkpoint)
    
    # Analyze scaling
    analysis = analyze_scaling(results_checkpoint)
    print("\n" + "=" * 90)
    print("SCALING ANALYSIS")
    print("=" * 90)
    print(f"Backprop scaling: {analysis.get('bp_scaling_mb_per_layer', 0):.4f} MB/layer")
    print(f"EP scaling:       {analysis.get('ep_scaling_mb_per_layer', 0):.4f} MB/layer")
    print(f"Memory savings at max depth: {analysis.get('memory_savings_at_max_depth_pct', 0):.1f}%")
    
    # Save results
    save_results(results_checkpoint, "memory_scaling_results_checkpoint.json")
    
    # Plot if possible
    try:
        plot_results(results_checkpoint, save_path="memory_scaling_plot.png")
    except Exception as e:
        print(f"\nCould not plot: {e}")
    
    # Conclusion
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    
    bp_scaling = analysis.get('bp_scaling_mb_per_layer', 0)
    ep_scaling = analysis.get('ep_scaling_mb_per_layer', 0)
    savings = analysis.get('memory_savings_at_max_depth_pct', 0)
    
    if bp_scaling > ep_scaling * 2:
        print(f"  ✅ EP shows O(1) scaling (backprop scales {bp_scaling/ep_scaling:.1f}x faster)")
        print("  EP's activation memory advantage is CONFIRMED at depth.")
    elif savings > 30:
        print(f"  ✅ EP uses {savings:.1f}% less activation memory at max depth")
        print("  EP's memory advantage is SIGNIFICANT.")
    elif savings > 10:
        print(f"  ⚠️  EP uses {savings:.1f}% less activation memory at max depth")
        print("  EP has MARGINAL memory advantage.")
    else:
        print(f"  ❌ EP uses {savings:.1f}% memory (no significant savings)")
        print("  EP's O(1) claim may not translate to practice with checkpointing.")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
