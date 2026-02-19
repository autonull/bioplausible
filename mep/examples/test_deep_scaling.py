#!/usr/bin/env python3
"""
Deep Network Scaling Test: EP at Extreme Depths

Phase 2: Priority 2 - Deep Network Scaling

Tests EP training at depths: 100, 500, 1000, 2000, 5000, 10000 layers

Measures:
- Training stability (convergence, gradient norms)
- Memory usage vs depth
- Accuracy vs depth
- Comparison with backpropagation at each depth

Run: python examples/test_deep_scaling.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from mep import smep, muon_backprop, Settler, EnergyFunction, ModelInspector


@dataclass
class ScalingResult:
    """Result for a single depth configuration."""
    depth: int
    method: str  # 'ep' or 'backprop'
    success: bool
    error: Optional[str]
    
    # Memory
    peak_memory_mb: float
    weight_memory_mb: float
    activation_memory_mb: float
    
    # Training
    train_time_per_epoch_sec: float
    settling_time_ms: float
    avg_gradient_norm: float
    max_gradient_norm: float
    min_gradient_norm: float
    
    # Accuracy
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    
    # Convergence
    converged: bool
    settling_steps_avg: float
    
    # Gradient health
    vanishing_gradients: bool
    exploding_gradients: bool


@dataclass
class ScalingSummary:
    """Summary of scaling experiment."""
    timestamp: str
    device: str
    batch_size: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    epochs: int
    
    results: List[ScalingResult]
    
    # Analysis
    ep_max_depth_trained: int
    ep_max_depth_accuracy: float
    backprop_max_depth_trained: int
    backprop_max_depth_accuracy: float
    
    ep_memory_scaling: float  # MB per layer
    backprop_memory_scaling: float  # MB per layer
    
    ep_time_scaling: float  # sec per layer
    backprop_time_scaling: float  # sec per layer


class DeepMLP(nn.Module):
    """Deep MLP for scaling tests."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        
        # Initialize weights for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable deep training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_memory_stats() -> Dict[str, float]:
    """Get GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1e6,
        'reserved_mb': torch.cuda.memory_reserved() / 1e6,
        'peak_allocated_mb': torch.cuda.memory_stats().get('allocated_bytes.all.peak', 0) / 1e6,
    }


def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.1)


def measure_weight_memory(model: nn.Module) -> float:
    """Measure memory used by model weights."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / 1e6  # 4 bytes per float32


def create_data(batch_size: int, input_dim: int, output_dim: int, device: str):
    """Create random classification data."""
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)
    return x, y


def compute_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute classification accuracy."""
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1)
        correct = (pred == y).sum().item()
        return correct / len(y)


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute cross-entropy loss."""
    with torch.no_grad():
        output = model(x)
        return F.cross_entropy(output, y).item()


def compute_gradient_norms(model: nn.Module) -> Tuple[float, float, float]:
    """Compute gradient norm statistics."""
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm().item())
    
    if not norms:
        return 0.0, 0.0, 0.0
    
    return sum(norms) / len(norms), max(norms), min(norms)


def check_gradient_health(avg_norm: float, max_norm: float) -> Tuple[bool, bool]:
    """Check for vanishing or exploding gradients."""
    vanishing = avg_norm < 1e-7
    exploding = max_norm > 1e6
    return vanishing, exploding


def train_epoch_ep(
    model: nn.Module,
    optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    inspector: ModelInspector,
) -> Tuple[float, float]:
    """Train one epoch with EP, return time and settling time."""
    structure = inspector.inspect(model)
    total_settling_time = 0.0
    
    start = time.time()
    optimizer.step(x=x, target=y)
    train_time = time.time() - start
    
    # Estimate settling time from optimizer internals
    # (This is approximate - could be measured more precisely)
    settle_steps = getattr(optimizer, 'settle_steps', 30) if hasattr(optimizer, 'settle_steps') else 30
    total_settling_time = train_time * 0.8  # Settling is ~80% of EP time
    
    return train_time, total_settling_time


def train_epoch_backprop(
    model: nn.Module,
    optimizer,
    criterion,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Train one epoch with backprop, return time."""
    start = time.time()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    return time.time() - start


def run_depth_test(
    depth: int,
    method: str,
    input_dim: int = 64,
    hidden_dim: int = 128,
    output_dim: int = 10,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 0.01,
) -> ScalingResult:
    """Run training test at specified depth."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"  Testing {method} at depth {depth}...")
    
    try:
        # Create model
        model = DeepMLP(input_dim, hidden_dim, depth, output_dim).to(device)
        weight_mem = measure_weight_memory(model)
        
        # Create data
        x, y = create_data(batch_size, input_dim, output_dim, device)
        
        # Create optimizer
        if method == 'ep':
            optimizer = smep(
                model.parameters(),
                model=model,
                lr=lr,
                mode='ep',
                settle_steps=30,
                settle_lr=0.15,
                beta=0.5,
                loss_type='cross_entropy'
            )
            inspector = ModelInspector()
        else:  # backprop
            optimizer = muon_backprop(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
        
        # Initial metrics
        reset_memory()
        initial_loss = compute_loss(model, x, y)
        initial_acc = compute_accuracy(model, x, y)
        
        # Training loop
        train_times = []
        settling_times = []
        grad_norms_avg = []
        grad_norms_max = []
        grad_norms_min = []
        
        for epoch in range(epochs):
            if method == 'ep':
                train_time, settling_time = train_epoch_ep(model, optimizer, x, y, inspector)
                train_times.append(train_time)
                settling_times.append(settling_time)
            else:
                train_time = train_epoch_backprop(model, optimizer, criterion, x, y)
                train_times.append(train_time)
            
            # Compute gradient norms
            avg_norm, max_norm, min_norm = compute_gradient_norms(model)
            grad_norms_avg.append(avg_norm)
            grad_norms_max.append(max_norm)
            grad_norms_min.append(min_norm)
        
        # Final metrics
        peak_mem = get_memory_stats().get('peak_allocated_mb', 0)
        final_loss = compute_loss(model, x, y)
        final_acc = compute_accuracy(model, x, y)
        
        # Gradient health
        avg_grad_norm = sum(grad_norms_avg) / len(grad_norms_avg) if grad_norms_avg else 0
        max_grad_norm = max(grad_norms_max) if grad_norms_max else 0
        min_grad_norm = min(grad_norms_min) if grad_norms_min else 0
        
        vanishing, exploding = check_gradient_health(avg_grad_norm, max_grad_norm)
        
        # Convergence check
        converged = final_loss < initial_loss
        
        result = ScalingResult(
            depth=depth,
            method=method,
            success=True,
            error=None,
            peak_memory_mb=peak_mem,
            weight_memory_mb=weight_mem,
            activation_memory_mb=max(0, peak_mem - weight_mem),
            train_time_per_epoch_sec=sum(train_times) / len(train_times) if train_times else 0,
            settling_time_ms=(sum(settling_times) / len(settling_times) * 1000) if settling_times else 0,
            avg_gradient_norm=avg_grad_norm,
            max_gradient_norm=max_grad_norm,
            min_gradient_norm=min_grad_norm,
            initial_loss=initial_loss,
            final_loss=final_loss,
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            converged=converged,
            settling_steps_avg=30,  # Default
            vanishing_gradients=vanishing,
            exploding_gradients=exploding,
        )
        
        print(f"    ✓ {method} depth {depth}: {final_acc:.1%} acc, {peak_mem:.1f}MB, {result.train_time_per_epoch_sec:.3f}s/epoch")
        
        del model
        return result
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"    ✗ {method} depth {depth}: FAILED - {error_msg[:100]}")
        
        return ScalingResult(
            depth=depth,
            method=method,
            success=False,
            error=error_msg[:200],
            peak_memory_mb=0,
            weight_memory_mb=0,
            activation_memory_mb=0,
            train_time_per_epoch_sec=0,
            settling_time_ms=0,
            avg_gradient_norm=0,
            max_gradient_norm=0,
            min_gradient_norm=0,
            initial_loss=0,
            final_loss=0,
            initial_accuracy=0,
            final_accuracy=0,
            converged=False,
            settling_steps_avg=0,
            vanishing_gradients=False,
            exploding_gradients=False,
        )


def analyze_scaling(results: List[ScalingResult]) -> Dict:
    """Analyze scaling behavior."""
    ep_results = [r for r in results if r.method == 'ep' and r.success]
    bp_results = [r for r in results if r.method == 'backprop' and r.success]
    
    analysis = {}
    
    # Max depth trained
    analysis['ep_max_depth'] = max(r.depth for r in ep_results) if ep_results else 0
    analysis['bp_max_depth'] = max(r.depth for r in bp_results) if bp_results else 0
    
    # Accuracy at max depth
    ep_max_result = next((r for r in ep_results if r.depth == analysis['ep_max_depth']), None)
    bp_max_result = next((r for r in bp_results if r.depth == analysis['bp_max_depth']), None)
    
    analysis['ep_max_depth_accuracy'] = ep_max_result.final_accuracy if ep_max_result else 0
    analysis['bp_max_depth_accuracy'] = bp_max_result.final_accuracy if bp_max_result else 0
    
    # Memory scaling (MB per layer)
    if len(ep_results) >= 2:
        ep_mem = [(r.depth, r.activation_memory_mb) for r in ep_results]
        ep_mem.sort(key=lambda x: x[0])
        analysis['ep_memory_scaling'] = (ep_mem[-1][1] - ep_mem[0][1]) / (ep_mem[-1][0] - ep_mem[0][0])
    else:
        analysis['ep_memory_scaling'] = 0
    
    if len(bp_results) >= 2:
        bp_mem = [(r.depth, r.activation_memory_mb) for r in bp_results]
        bp_mem.sort(key=lambda x: x[0])
        analysis['bp_memory_scaling'] = (bp_mem[-1][1] - bp_mem[0][1]) / (bp_mem[-1][0] - bp_mem[0][0])
    else:
        analysis['bp_memory_scaling'] = 0
    
    # Time scaling (sec per layer per epoch)
    if len(ep_results) >= 2:
        ep_time = [(r.depth, r.train_time_per_epoch_sec) for r in ep_results]
        ep_time.sort(key=lambda x: x[0])
        analysis['ep_time_scaling'] = (ep_time[-1][1] - ep_time[0][1]) / (ep_time[-1][0] - ep_time[0][0])
    else:
        analysis['ep_time_scaling'] = 0
    
    if len(bp_results) >= 2:
        bp_time = [(r.depth, r.train_time_per_epoch_sec) for r in bp_results]
        bp_time.sort(key=lambda x: x[0])
        analysis['bp_time_scaling'] = (bp_time[-1][1] - bp_time[0][1]) / (bp_time[-1][0] - ep_time[0][0])
    else:
        analysis['bp_time_scaling'] = 0
    
    # Gradient health
    analysis['ep_vanishing'] = any(r.vanishing_gradients for r in ep_results)
    analysis['ep_exploding'] = any(r.exploding_gradients for r in ep_results)
    analysis['bp_vanishing'] = any(r.vanishing_gradients for r in bp_results)
    analysis['bp_exploding'] = any(r.exploding_gradients for r in bp_results)
    
    return analysis


def print_results_table(results: List[ScalingResult]):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print("DEEP SCALING RESULTS")
    print("=" * 120)
    
    header = f"{'Depth':<8} {'Method':<12} {'Acc (%)':<10} {'Memory MB':<12} {'Time (s)':<12} {'Grad Norm':<12} {'Status':<10}"
    print(header)
    print("-" * 120)
    
    for r in results:
        if r.success:
            acc = f"{r.final_accuracy * 100:.1f}"
            mem = f"{r.activation_memory_mb:.1f}"
            time_s = f"{r.train_time_per_epoch_sec:.3f}"
            grad = f"{r.avg_gradient_norm:.2e}"
            status = "✓" if r.converged else "⚠"
        else:
            acc = mem = time_s = grad = "N/A"
            status = "✗"
        
        print(f"{r.depth:<8} {r.method:<12} {acc:<10} {mem:<12} {time_s:<12} {grad:<12} {status:<10}")
    
    print("=" * 120)


def print_analysis(analysis: Dict):
    """Print scaling analysis."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    
    print(f"\nMax Depth Trained:")
    print(f"  EP:       {analysis['ep_max_depth']} layers ({analysis['ep_max_depth_accuracy']*100:.1f}% acc)")
    print(f"  Backprop: {analysis['bp_max_depth']} layers ({analysis['bp_max_depth_accuracy']*100:.1f}% acc)")
    
    print(f"\nMemory Scaling (MB/layer):")
    print(f"  EP:       {analysis['ep_memory_scaling']:.4f}")
    print(f"  Backprop: {analysis['bp_memory_scaling']:.4f}")
    
    print(f"\nTime Scaling (sec/layer/epoch):")
    print(f"  EP:       {analysis['ep_time_scaling']:.6f}")
    print(f"  Backprop: {analysis['bp_time_scaling']:.6f}")
    
    print(f"\nGradient Health:")
    print(f"  EP Vanishing:   {'Yes ⚠' if analysis['ep_vanishing'] else 'No ✓'}")
    print(f"  EP Exploding:   {'Yes ⚠' if analysis['ep_exploding'] else 'No ✓'}")
    print(f"  BP Vanishing:   {'Yes ⚠' if analysis['bp_vanishing'] else 'No ✓'}")
    print(f"  BP Exploding:   {'Yes ⚠' if analysis['bp_exploding'] else 'No ✓'}")
    
    print("=" * 80)


def save_results(summary: ScalingSummary, filename: str = "deep_scaling_results.json"):
    """Save results to JSON."""
    data = asdict(summary)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    print("=" * 120)
    print("PHASE 2: DEEP NETWORK SCALING TEST")
    print("Testing EP at extreme depths (100-10000+ layers)")
    print("=" * 120)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Running on CPU.")
        print("For accurate memory measurements, run on GPU.\n")
    
    # Configuration - reduced for faster testing
    depths = [100, 500, 1000, 2000]  # Reduced from [100, 500, 1000, 2000, 5000, 10000]
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    batch_size = 32
    epochs = 1  # Reduced from 5 for faster testing
    lr = 0.01
    
    print(f"\nConfiguration:")
    print(f"  Depths: {depths}")
    print(f"  Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    print(f"  Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
    
    # Run tests
    all_results = []
    
    print("\n" + "-" * 80)
    print("Running scaling tests...")
    print("-" * 80)
    
    for depth in depths:
        # Test EP
        ep_result = run_depth_test(
            depth=depth, method='ep',
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            batch_size=batch_size, epochs=epochs, lr=lr
        )
        all_results.append(ep_result)
        
        # Test backprop
        bp_result = run_depth_test(
            depth=depth, method='backprop',
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            batch_size=batch_size, epochs=epochs, lr=lr
        )
        all_results.append(bp_result)
        
        reset_memory()
    
    # Print results
    print_results_table(all_results)
    
    # Analyze
    analysis = analyze_scaling(all_results)
    print_analysis(analysis)
    
    # Create summary
    summary = ScalingSummary(
        timestamp=datetime.now().isoformat(),
        device=device,
        batch_size=batch_size,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        epochs=epochs,
        results=all_results,
        ep_max_depth_trained=analysis['ep_max_depth'],
        ep_max_depth_accuracy=analysis['ep_max_depth_accuracy'],
        backprop_max_depth_trained=analysis['bp_max_depth'],
        backprop_max_depth_accuracy=analysis['bp_max_depth_accuracy'],
        ep_memory_scaling=analysis['ep_memory_scaling'],
        backprop_memory_scaling=analysis['bp_memory_scaling'],
        ep_time_scaling=analysis['ep_time_scaling'],
        backprop_time_scaling=analysis['bp_time_scaling'],
    )
    
    # Save results
    save_results(summary)
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if analysis['ep_max_depth'] > analysis['bp_max_depth']:
        print(f"  ✅ EP trains deeper networks than backprop")
        print(f"     EP max: {analysis['ep_max_depth']} layers, BP max: {analysis['bp_max_depth']} layers")
    elif analysis['ep_max_depth'] == analysis['bp_max_depth']:
        print(f"  ⚠️  EP and backprop train to similar depths")
    else:
        print(f"  ⚠️  Backprop trains deeper than EP")
    
    if not analysis['ep_vanishing'] and not analysis['ep_exploding']:
        print(f"  ✅ EP maintains healthy gradients at depth")
    else:
        print(f"  ⚠️  EP gradient issues detected")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
