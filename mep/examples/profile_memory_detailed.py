#!/usr/bin/env python3
"""
Detailed Memory Profiling for EP vs Backpropagation

Phase 2: Week 1-2 - Memory profiling and baseline establishment

Measures:
- Activation memory (excluding weights)
- Memory by component (settling, energy, contrast)
- Memory vs depth scaling
- PyTorch operations triggering activation storage

Run: python examples/profile_memory_detailed.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json

from mep import smep, muon_backprop, Settler, EnergyFunction, ModelInspector


@dataclass
class MemoryMeasurement:
    """Memory measurement for a single configuration."""
    depth: int
    method: str
    activation_memory_mb: float
    total_memory_mb: float
    weight_memory_mb: float
    success: bool
    error: Optional[str] = None
    train_time_sec: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0


@dataclass
class ComponentProfile:
    """Memory profile by EP component."""
    depth: int
    settling_memory_mb: float
    energy_memory_mb: float
    contrast_memory_mb: float
    total_activation_mb: float
    settling_time_ms: float
    energy_time_ms: float
    contrast_time_ms: float


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
    time.sleep(0.1)


def measure_weight_memory(model: nn.Module) -> float:
    """Measure memory used by model weights only."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assume 4 bytes per float32 parameter
    return total_params * 4 / 1e6


class DeepMLP(nn.Module):
    """Deep MLP with configurable depth for memory testing."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

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


def make_deep_mlp(input_dim: int = 784, hidden_dim: int = 128, 
                  num_layers: int = 100, output_dim: int = 10,
                  device: str = 'cuda') -> nn.Module:
    """Create a deep MLP for testing."""
    model = DeepMLP(input_dim, hidden_dim, num_layers, output_dim).to(device)
    return model


def measure_activation_memory(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str = 'ep',
    lr: float = 0.01,
) -> MemoryMeasurement:
    """
    Measure activation memory for a single training step.
    
    Key: We measure the PEAK memory during training, then subtract
    weight memory to isolate activation memory.
    """
    reset_memory()
    
    # Get baseline memory (model weights only)
    weight_mem = measure_weight_memory(model)
    
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
            settle_steps=30,
            settle_lr=0.15,
            loss_type='cross_entropy',
        )
    
    # Training step
    start = time.time()
    
    if method == 'backprop':
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    else:  # EP
        optimizer.step(x=x, target=y)
    
    train_time = time.time() - start
    
    # Get peak memory
    mem_stats = get_memory_stats()
    peak_allocated = mem_stats.get('peak_allocated_mb', 0)
    activation_mem = peak_allocated - weight_mem
    
    return MemoryMeasurement(
        depth=model.num_layers,
        method=method,
        activation_memory_mb=max(0, activation_mem),
        total_memory_mb=peak_allocated,
        weight_memory_mb=weight_mem,
        success=True,
        train_time_sec=train_time,
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=mem_stats.get('peak_reserved_mb', 0),
    )


def run_baseline_comparison(
    depths: List[int],
    input_dim: int = 784,
    hidden_dim: int = 128,
    output_dim: int = 10,
    batch_size: int = 32,
) -> Tuple[List[MemoryMeasurement], List[MemoryMeasurement]]:
    """Run baseline memory comparison for all depths."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    results_bp = []
    results_ep = []
    
    # Create data
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)
    
    for depth in depths:
        print(f"\nTesting depth {depth}...")
        
        # Test backprop
        model_bp = make_deep_mlp(input_dim, hidden_dim, depth, output_dim, device)
        
        try:
            result_bp = measure_activation_memory(model_bp, x, y, method='backprop')
            results_bp.append(result_bp)
            print(f"  Backprop: {result_bp.activation_memory_mb:.2f} MB (activation), "
                  f"{result_bp.total_memory_mb:.2f} MB (total), "
                  f"{result_bp.train_time_sec:.3f}s")
        except RuntimeError as e:
            result_bp = MemoryMeasurement(
                depth=depth, method='backprop',
                activation_memory_mb=0, total_memory_mb=0, weight_memory_mb=0,
                success=False, error=str(e)
            )
            results_bp.append(result_bp)
            print(f"  Backprop: FAILED - {e}")
        
        del model_bp
        reset_memory()
        
        # Test EP
        model_ep = make_deep_mlp(input_dim, hidden_dim, depth, output_dim, device)
        
        try:
            result_ep = measure_activation_memory(model_ep, x, y, method='ep')
            results_ep.append(result_ep)
            print(f"  EP:       {result_ep.activation_memory_mb:.2f} MB (activation), "
                  f"{result_ep.total_memory_mb:.2f} MB (total), "
                  f"{result_ep.train_time_sec:.3f}s")
        except RuntimeError as e:
            result_ep = MemoryMeasurement(
                depth=depth, method='ep',
                activation_memory_mb=0, total_memory_mb=0, weight_memory_mb=0,
                success=False, error=str(e)
            )
            results_ep.append(result_ep)
            print(f"  EP:       FAILED - {e}")
        
        del model_ep
        reset_memory()
    
    return results_bp, results_ep


def profile_by_component(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    inspector: ModelInspector,
    structure: List[Dict[str, Any]],
    settler: Settler,
    energy_fn: EnergyFunction,
) -> ComponentProfile:
    """
    Profile memory usage by EP component.
    
    Components:
    - Settling loop (free phase + nudged phase)
    - Energy computation
    - Contrast step (gradient computation)
    """
    device = x.device
    batch_size = x.shape[0]
    
    # Prepare target
    target_vec = y
    
    beta = 0.5
    
    # Profile settling (free phase)
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_free:
        states_free = settler.settle(model, x, None, beta=0.0, energy_fn=energy_fn, structure=structure)
    settling_time_free = (time.time() - start) * 1000
    settling_mem_free = torch.cuda.max_memory_allocated() / 1e6
    
    # Profile settling (nudged phase)
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_nudged:
        states_nudged = settler.settle(model, x, target_vec, beta=beta, energy_fn=energy_fn, structure=structure)
    settling_time_nudged = (time.time() - start) * 1000
    settling_mem_nudged = torch.cuda.max_memory_allocated() / 1e6
    
    # Total settling
    settling_memory_mb = max(settling_mem_free, settling_mem_nudged)
    settling_time_ms = settling_time_free + settling_time_nudged
    
    # Profile energy computation
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_energy:
        E_free = energy_fn(model, x, states_free, structure, None, 0.0)
        E_nudged = energy_fn(model, x, states_nudged, structure, target_vec, beta)
    energy_time_ms = (time.time() - start) * 1000
    energy_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    
    # Profile contrast step
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_contrast:
        contrast_loss = (E_nudged - E_free) / beta
        params = list(model.parameters())
        grads = torch.autograd.grad(contrast_loss, params, retain_graph=False)
    contrast_time_ms = (time.time() - start) * 1000
    contrast_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    
    # Total activation memory
    total_activation_mb = settling_memory_mb + energy_memory_mb + contrast_memory_mb
    
    return ComponentProfile(
        depth=model.num_layers,
        settling_memory_mb=settling_memory_mb,
        energy_memory_mb=energy_memory_mb,
        contrast_memory_mb=contrast_memory_mb,
        total_activation_mb=total_activation_mb,
        settling_time_ms=settling_time_ms,
        energy_time_ms=energy_time_ms,
        contrast_time_ms=contrast_time_ms,
    )


def analyze_pytorch_operations(
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[str, float]:
    """
    Identify which PyTorch operations trigger activation storage.
    
    Tests:
    1. Standard forward (with autograd)
    2. no_grad forward
    3. Manual forward (direct matmul)
    """
    results = {}
    
    # Test 1: Standard forward (with autograd)
    reset_memory()
    with torch.enable_grad():
        h = model(x)
    results['with_grad_mb'] = torch.cuda.memory_allocated() / 1e6
    
    # Test 2: no_grad forward
    reset_memory()
    with torch.no_grad():
        h = model(x)
    results['no_grad_mb'] = torch.cuda.memory_allocated() / 1e6
    
    # Test 3: Manual forward (layer by layer)
    reset_memory()
    with torch.no_grad():
        h = x
        for module in model.network:
            if isinstance(module, nn.Linear):
                # Manual computation
                h = h @ module.weight.t() + module.bias
            elif isinstance(module, nn.ReLU):
                h = F.relu(h)
    results['manual_mb'] = torch.cuda.memory_allocated() / 1e6
    
    return results


def print_baseline_results(
    bp_results: List[MemoryMeasurement],
    ep_results: List[MemoryMeasurement],
):
    """Print formatted baseline results table."""
    print("\n" + "=" * 100)
    print("BASELINE MEMORY SCALING RESULTS")
    print("=" * 100)
    print(f"{'Depth':<8} {'Method':<12} {'Activation MB':<18} {'Total MB':<15} {'Time (s)':<12} {'Status':<8}")
    print("-" * 100)
    
    all_results = []
    for bp, ep in zip(bp_results, ep_results):
        all_results.append((bp.depth, bp, ep))
    
    all_results.sort(key=lambda x: x[0])
    
    for depth, bp, ep in all_results:
        bp_act = f"{bp.activation_memory_mb:.2f}" if bp.success else "N/A"
        bp_tot = f"{bp.total_memory_mb:.2f}" if bp.success else "N/A"
        bp_time = f"{bp.train_time_sec:.3f}" if bp.success else "N/A"
        bp_status = "✓" if bp.success else "✗"
        
        ep_act = f"{ep.activation_memory_mb:.2f}" if ep.success else "N/A"
        ep_tot = f"{ep.total_memory_mb:.2f}" if ep.success else "N/A"
        ep_time = f"{ep.train_time_sec:.3f}" if ep.success else "N/A"
        ep_status = "✓" if ep.success else "✗"
        
        print(f"{depth:<8} {bp.method:<12} {bp_act:<18} {bp_tot:<15} {bp_time:<12} {bp_status:<8}")
        print(f"{'':<8} {ep.method:<12} {ep_act:<18} {ep_tot:<15} {ep_time:<12} {ep_status:<8}")
        print("-" * 100)
    
    # Calculate scaling
    successful_bp = [r for r in bp_results if r.success]
    successful_ep = [r for r in ep_results if r.success]
    
    if len(successful_bp) >= 2 and len(successful_ep) >= 2:
        bp_scaling = (successful_bp[-1].activation_memory_mb - successful_bp[0].activation_memory_mb) / \
                     (successful_bp[-1].depth - successful_bp[0].depth)
        ep_scaling = (successful_ep[-1].activation_memory_mb - successful_ep[0].activation_memory_mb) / \
                     (successful_ep[-1].depth - successful_ep[0].depth)
        
        print(f"\nScaling Analysis:")
        print(f"  Backprop: {bp_scaling:.4f} MB/layer")
        print(f"  EP:       {ep_scaling:.4f} MB/layer")
        
        if bp_scaling > 0:
            savings_at_max = (1 - successful_ep[-1].activation_memory_mb / successful_bp[-1].activation_memory_mb) * 100
            print(f"  Savings at depth {successful_ep[-1].depth}: {savings_at_max:.1f}%")


def print_component_profile(profile: ComponentProfile):
    """Print component profile results."""
    print("\n" + "=" * 80)
    print(f"COMPONENT MEMORY PROFILE (Depth {profile.depth})")
    print("=" * 80)
    print(f"{'Component':<20} {'Memory (MB)':<18} {'Time (ms)':<15} {'Fraction':<10}")
    print("-" * 80)
    
    total = profile.total_activation_mb
    if total > 0:
        settling_frac = profile.settling_memory_mb / total * 100
        energy_frac = profile.energy_memory_mb / total * 100
        contrast_frac = profile.contrast_memory_mb / total * 100
    else:
        settling_frac = energy_frac = contrast_frac = 0
    
    print(f"{'Settling':<20} {profile.settling_memory_mb:<18.2f} {profile.settling_time_ms:<15.2f} {settling_frac:>5.1f}%")
    print(f"{'Energy':<20} {profile.energy_memory_mb:<18.2f} {profile.energy_time_ms:<15.2f} {energy_frac:>5.1f}%")
    print(f"{'Contrast':<20} {profile.contrast_memory_mb:<18.2f} {profile.contrast_time_ms:<15.2f} {contrast_frac:>5.1f}%")
    print("-" * 80)
    print(f"{'Total':<20} {total:<18.2f} {profile.settling_time_ms + profile.energy_time_ms + profile.contrast_time_ms:<15.2f}")
    print("=" * 80)


def print_pytorch_analysis(analysis: Dict[str, float]):
    """Print PyTorch operation analysis."""
    print("\n" + "=" * 80)
    print("PYTORCH OPERATION ANALYSIS")
    print("=" * 80)
    print(f"{'Operation':<30} {'Memory (MB)':<18} {'Overhead':<10}")
    print("-" * 80)
    
    baseline = analysis['manual_mb']
    with_grad = analysis['with_grad_mb']
    no_grad = analysis['no_grad_mb']
    
    print(f"{'Manual (no_grad)':<30} {baseline:<18.2f} {'(baseline)':<10}")
    print(f"{'nn.Module (no_grad)':<30} {no_grad:<18.2f} {no_grad - baseline:>9.2f} MB")
    print(f"{'nn.Module (enable_grad)':<30} {with_grad:<18.2f} {with_grad - baseline:>9.2f} MB")
    print("=" * 80)
    
    print("\nKey findings:")
    if with_grad - no_grad > 1.0:
        print(f"  ⚠️  enable_grad() adds {with_grad - no_grad:.2f} MB overhead (activation storage)")
    if no_grad - baseline > 1.0:
        print(f"  ⚠️  nn.Module dispatch adds {no_grad - baseline:.2f} MB overhead")


def save_results(
    bp_results: List[MemoryMeasurement],
    ep_results: List[MemoryMeasurement],
    component_profile: Optional[ComponentProfile] = None,
    pytorch_analysis: Optional[Dict[str, float]] = None,
):
    """Save results to JSON file."""
    data = {
        'baseline_bp': [asdict(r) for r in bp_results],
        'baseline_ep': [asdict(r) for r in ep_results],
    }
    
    if component_profile:
        data['component_profile'] = asdict(component_profile)
    
    if pytorch_analysis:
        data['pytorch_analysis'] = pytorch_analysis
    
    with open("memory_profile_results.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: memory_profile_results.json")


def main():
    print("=" * 100)
    print("PHASE 2: WEEK 1-2 - DETAILED MEMORY PROFILING")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Running on CPU.")
        print("Memory measurements will be less meaningful.")
        print("For accurate results, run on GPU.\n")
    
    # Test depths
    depths = [10, 50, 100, 200, 500, 1000]
    
    print("\n" + "=" * 100)
    print("TASK 1: BASELINE MEMORY VS DEPTH")
    print("=" * 100)
    
    bp_results, ep_results = run_baseline_comparison(depths=depths)
    print_baseline_results(bp_results, ep_results)
    
    # Component profiling at representative depth
    print("\n" + "=" * 100)
    print("TASK 2: COMPONENT MEMORY PROFILE")
    print("=" * 100)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_deep_mlp(num_layers=100, device=device)
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    settler = Settler(steps=30, lr=0.15)
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    
    component_profile = profile_by_component(model, x, y, inspector, structure, settler, energy_fn)
    print_component_profile(component_profile)
    
    # PyTorch operation analysis
    print("\n" + "=" * 100)
    print("TASK 3: PYTORCH OPERATION ANALYSIS")
    print("=" * 100)
    
    pytorch_analysis = analyze_pytorch_operations(model, x)
    print_pytorch_analysis(pytorch_analysis)
    
    # Save results
    save_results(bp_results, ep_results, component_profile, pytorch_analysis)
    
    # Conclusion
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    
    successful_bp = [r for r in bp_results if r.success]
    successful_ep = [r for r in ep_results if r.success]
    
    if len(successful_bp) >= 2 and len(successful_ep) >= 2:
        bp_scaling = (successful_bp[-1].activation_memory_mb - successful_bp[0].activation_memory_mb) / \
                     (successful_bp[-1].depth - successful_bp[0].depth)
        ep_scaling = (successful_ep[-1].activation_memory_mb - successful_ep[0].activation_memory_mb) / \
                     (successful_ep[-1].depth - successful_ep[0].depth)
        
        if bp_scaling > ep_scaling * 2:
            print(f"  ✅ EP shows better scaling (backprop scales {bp_scaling/ep_scaling:.1f}x faster)")
        else:
            print(f"  ⚠️  EP scaling similar to backprop (both store activations)")
        
        print(f"\n  Next steps:")
        print(f"  1. Implement manual settling without autograd")
        print(f"  2. Implement no-grad energy computation")
        print(f"  3. Target: 50%+ memory savings at depth 500")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
