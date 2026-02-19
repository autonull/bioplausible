#!/usr/bin/env python3
"""
O(1) Memory v2: Analytic Gradients Verification

Phase 2: Week 3-4 - True O(1) Memory via Analytic Gradients

Tests:
1. Analytic gradients match autograd gradients (<1e-5 difference)
2. Analytic settling matches current settling (<1e-5 difference)
3. Memory savings: 50%+ at depth 500

Run: python examples/verify_o1_memory_v2.py
"""

import torch
import torch.nn as nn
import gc
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from mep import smep, Settler, EnergyFunction, ModelInspector
from mep.optimizers import (
    analytic_state_gradients,
    settle_manual_o1,
    manual_energy_compute_o1,
    O1MemoryEPv2,
)


@dataclass
class CorrectnessResult:
    """Correctness verification result."""
    test_name: str
    passed: bool
    difference: float
    tolerance: float
    details: str = ""


@dataclass
class MemoryComparison:
    """Memory comparison result."""
    depth: int
    current_ep_mb: float
    o1_ep_v2_mb: float
    savings_mb: float
    savings_percent: float
    current_time_s: float
    o1_time_s: float


def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.1)


def get_peak_memory_mb() -> float:
    """Get peak allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_stats().get('allocated_bytes.all.peak', 0) / 1e6
    return 0.0


class DeepMLP(nn.Module):
    """Deep MLP for testing."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_model(input_dim: int = 784, hidden_dim: int = 128, 
               num_layers: int = 100, output_dim: int = 10,
               device: str = 'cuda') -> nn.Module:
    """Create a deep MLP for testing."""
    return DeepMLP(input_dim, hidden_dim, num_layers, output_dim).to(device)


def verify_analytic_gradients(
    model: nn.Module,
    x: torch.Tensor,
    states: List[torch.Tensor],
    structure: List[Dict[str, Any]],
    target_vec: torch.Tensor,
    beta: float,
) -> CorrectnessResult:
    """
    Verify analytic gradients match autograd gradients.
    
    Success criteria: Mean gradient difference < 1e-5
    """
    # Compute gradients using autograd (reference)
    states_for_autograd = [s.detach().clone().requires_grad_(True) for s in states]
    
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    E_autograd = energy_fn(model, x, states_for_autograd, structure, target_vec, beta)
    
    grads_autograd = torch.autograd.grad(E_autograd, states_for_autograd, retain_graph=False)
    
    # Compute gradients analytically
    grads_analytic = analytic_state_gradients(
        model, x, states, structure, target_vec, beta,
        loss_type='cross_entropy'
    )
    
    # Compare gradients
    if len(grads_autograd) != len(grads_analytic):
        return CorrectnessResult(
            test_name="Analytic Gradients",
            passed=False,
            difference=float('inf'),
            tolerance=1e-5,
            details=f"Gradient count mismatch: {len(grads_autograd)} vs {len(grads_analytic)}"
        )
    
    max_diff = 0.0
    total_diff = 0.0
    count = 0
    
    for g_auto, g_analytic in zip(grads_autograd, grads_analytic):
        diff_tensor = (g_auto - g_analytic).abs()
        max_diff = max(max_diff, diff_tensor.max().item())
        total_diff += diff_tensor.mean().item()
        count += 1
    
    mean_diff = total_diff / count if count > 0 else float('inf')
    tolerance = 1e-5
    
    passed = mean_diff < tolerance
    
    return CorrectnessResult(
        test_name="Analytic Gradients",
        passed=passed,
        difference=mean_diff,
        tolerance=tolerance,
        details=f"Mean diff={mean_diff:.6e}, Max diff={max_diff:.6e}"
    )


def verify_settle_analytic(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    structure: List[Dict[str, Any]],
    beta: float = 0.5,
    steps: int = 30,
    lr: float = 0.15,
) -> CorrectnessResult:
    """
    Verify analytic settling matches current implementation.
    
    Success criteria: Mean state difference < 1e-5
    """
    settler = Settler(steps=steps, lr=lr)
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    
    # Current settling (with autograd)
    states_current = settler.settle(
        model, x, target, beta=beta,
        energy_fn=energy_fn, structure=structure
    )
    
    # Analytic settling (no autograd)
    states_analytic = settle_manual_o1(
        model, x, target, beta=beta,
        structure=structure,
        steps=steps,
        lr=lr,
        loss_type='cross_entropy'
    )
    
    # Compare states
    if len(states_current) != len(states_analytic):
        return CorrectnessResult(
            test_name="Analytic Settling",
            passed=False,
            difference=float('inf'),
            tolerance=1e-5,
            details=f"State count mismatch: {len(states_current)} vs {len(states_analytic)}"
        )
    
    max_diff = 0.0
    total_diff = 0.0
    count = 0
    
    for s1, s2 in zip(states_current, states_analytic):
        diff_tensor = (s1 - s2).abs()
        max_diff = max(max_diff, diff_tensor.max().item())
        total_diff += diff_tensor.mean().item()
        count += 1
    
    mean_diff = total_diff / count if count > 0 else float('inf')
    tolerance = 1e-5
    
    passed = mean_diff < tolerance
    
    return CorrectnessResult(
        test_name="Analytic Settling",
        passed=passed,
        difference=mean_diff,
        tolerance=tolerance,
        details=f"Mean diff={mean_diff:.6e}, Max diff={max_diff:.6e}"
    )


def verify_o1_v2_training(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    structure: List[Dict[str, Any]],
) -> Tuple[CorrectnessResult, float, float]:
    """
    Verify O(1) v2 training produces similar loss.
    
    Success criteria: Loss difference < 1e-3 after one step
    """
    # Current EP
    model_current = model
    optimizer_current = smep(
        model_current.parameters(),
        model=model_current,
        lr=0.01,
        mode='ep',
        settle_steps=30,
        settle_lr=0.15,
        beta=0.5,
        loss_type='cross_entropy'
    )
    
    reset_memory()
    
    # Current EP step
    start = time.time()
    optimizer_current.step(x=x, target=y)
    time_current = time.time() - start
    
    # Compute loss after step
    with torch.no_grad():
        output_current = model_current(x)
        loss_current = nn.functional.cross_entropy(output_current, y).item()
    
    # O(1) v2 EP
    model_o1 = DeepMLP(x.shape[1], model.hidden_dim, model.num_layers, y.max().item() + 1)
    model_o1.load_state_dict(model.state_dict())
    model_o1.to(x.device)
    
    optimizer_o1 = O1MemoryEPv2(
        model_o1.parameters(),
        model=model_o1,
        lr=0.01,
        settle_steps=30,
        settle_lr=0.15,
        beta=0.5,
        loss_type='cross_entropy'
    )
    
    reset_memory()
    
    # O(1) v2 EP step
    start = time.time()
    optimizer_o1.step(x=x, target=y)
    time_o1 = time.time() - start
    
    # Compute loss after step
    with torch.no_grad():
        output_o1 = model_o1(x)
        loss_o1 = nn.functional.cross_entropy(output_o1, y).item()
    
    # Compare losses
    diff = abs(loss_current - loss_o1)
    tolerance = 1e-3
    
    passed = diff < tolerance
    
    result = CorrectnessResult(
        test_name="O(1) v2 Training Step",
        passed=passed,
        difference=diff,
        tolerance=tolerance,
        details=f"Loss current={loss_current:.6f}, Loss O(1) v2={loss_o1:.6f}"
    )
    
    return result, time_current, time_o1


def measure_memory_savings_v2(
    depth: int,
    input_dim: int = 784,
    hidden_dim: int = 128,
    output_dim: int = 10,
    batch_size: int = 32,
) -> MemoryComparison:
    """
    Measure memory savings of O(1) v2 implementation vs current EP.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create data
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)
    
    # Current EP
    model_current = make_model(input_dim, hidden_dim, depth, output_dim, device)
    optimizer_current = smep(
        model_current.parameters(),
        model=model_current,
        lr=0.01,
        mode='ep',
        settle_steps=30,
        settle_lr=0.15,
        beta=0.5,
        loss_type='cross_entropy'
    )
    
    reset_memory()
    
    start = time.time()
    optimizer_current.step(x=x, target=y)
    time_current = time.time() - start
    
    mem_current = get_peak_memory_mb()
    
    del model_current, optimizer_current
    reset_memory()
    
    # O(1) v2 Memory EP
    model_o1 = make_model(input_dim, hidden_dim, depth, output_dim, device)
    optimizer_o1 = O1MemoryEPv2(
        model_o1.parameters(),
        model=model_o1,
        lr=0.01,
        settle_steps=30,
        settle_lr=0.15,
        beta=0.5,
        loss_type='cross_entropy'
    )
    
    reset_memory()
    
    start = time.time()
    optimizer_o1.step(x=x, target=y)
    time_o1 = time.time() - start
    
    mem_o1 = get_peak_memory_mb()
    
    savings_mb = mem_current - mem_o1
    savings_percent = (savings_mb / mem_current * 100) if mem_current > 0 else 0
    
    return MemoryComparison(
        depth=depth,
        current_ep_mb=mem_current,
        o1_ep_v2_mb=max(0, mem_o1),
        savings_mb=max(0, savings_mb),
        savings_percent=max(0, savings_percent),
        current_time_s=time_current,
        o1_time_s=time_o1
    )


def print_correctness_results(results: List[CorrectnessResult]):
    """Print correctness verification results."""
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION RESULTS")
    print("=" * 80)
    
    all_passed = True
    
    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        all_passed = all_passed and result.passed
        
        print(f"\n{status}: {result.test_name}")
        print(f"  Difference: {result.difference:.6e} (tolerance: {result.tolerance:.6e})")
        print(f"  Details: {result.details}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CORRECTNESS TESTS PASSED")
    else:
        print("❌ SOME CORRECTNESS TESTS FAILED")
    print("=" * 80)


def print_memory_results(results: List[MemoryComparison]):
    """Print memory comparison results."""
    print("\n" + "=" * 100)
    print("MEMORY SAVINGS: O(1) v2 EP vs Current EP")
    print("=" * 100)
    print(f"{'Depth':<8} {'Current EP (MB)':<18} {'O(1) v2 (MB)':<15} {'Savings (MB)':<15} {'Savings (%)':<12} {'Time Ratio':<10}")
    print("-" * 100)
    
    for r in results:
        time_ratio = r.o1_time_s / r.current_time_s if r.current_time_s > 0 else 0
        print(f"{r.depth:<8} {r.current_ep_mb:<18.2f} {r.o1_ep_v2_mb:<15.2f} {r.savings_mb:<15.2f} {r.savings_percent:>10.1f}% {time_ratio:>8.2f}x")
    
    print("=" * 100)
    
    # Summary
    if results:
        avg_savings = sum(r.savings_percent for r in results) / len(results)
        max_savings = max(r.savings_percent for r in results)
        
        print(f"\nSummary:")
        print(f"  Average savings: {avg_savings:.1f}%")
        print(f"  Maximum savings: {max_savings:.1f}% (depth {max(r.depth for r in results)})")
        
        # Check success criteria
        depth_500 = next((r for r in results if r.depth == 500), None)
        if depth_500:
            if depth_500.savings_percent >= 50:
                print(f"\n✅ SUCCESS: {depth_500.savings_percent:.1f}% savings at depth 500 (target: 50%+)")
            else:
                print(f"\n⚠️  BELOW TARGET: {depth_500.savings_percent:.1f}% savings at depth 500 (target: 50%+)")


def main():
    print("=" * 100)
    print("PHASE 2: WEEK 3-4 - O(1) MEMORY V2 (ANALYTIC GRADIENTS) VERIFICATION")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Running on CPU.")
        print("Memory measurements will be less meaningful.\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test model and data
    depth = 100
    model = make_model(num_layers=depth, device=device)
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    # Get settled states for gradient tests
    settler = Settler(steps=30, lr=0.15)
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    
    print("\n" + "=" * 100)
    print("TEST 1: ANALYTIC GRADIENTS CORRECTNESS")
    print("=" * 100)
    
    states_free = settler.settle(model, x, None, beta=0.0, energy_fn=energy_fn, structure=structure)
    grad_result = verify_analytic_gradients(model, x, states_free, structure, None, 0.0)
    
    print(f"\n{grad_result.test_name}")
    print(f"  Status: {'✓ PASS' if grad_result.passed else '✗ FAIL'}")
    print(f"  Difference: {grad_result.difference:.6e} (tolerance: {grad_result.tolerance:.6e})")
    print(f"  Details: {grad_result.details}")
    
    print("\n" + "=" * 100)
    print("TEST 2: ANALYTIC SETTLING CORRECTNESS")
    print("=" * 100)
    
    settle_result = verify_settle_analytic(model, x, y, structure, beta=0.5, steps=30, lr=0.15)
    
    print(f"\n{settle_result.test_name}")
    print(f"  Status: {'✓ PASS' if settle_result.passed else '✗ FAIL'}")
    print(f"  Difference: {settle_result.difference:.6e} (tolerance: {settle_result.tolerance:.6e})")
    print(f"  Details: {settle_result.details}")
    
    print("\n" + "=" * 100)
    print("TEST 3: O(1) V2 TRAINING CORRECTNESS")
    print("=" * 100)
    
    training_result, time_current, time_o1 = verify_o1_v2_training(model, x, y, structure)
    
    print(f"\n{training_result.test_name}")
    print(f"  Status: {'✓ PASS' if training_result.passed else '✗ FAIL'}")
    print(f"  Difference: {training_result.difference:.6e} (tolerance: {training_result.tolerance:.6e})")
    print(f"  Details: {training_result.details}")
    print(f"  Time: Current={time_current:.3f}s, O(1) v2={time_o1:.3f}s ({time_o1/time_current:.2f}x)")
    
    # Print all correctness results
    print_correctness_results([grad_result, settle_result, training_result])
    
    # Memory savings at multiple depths
    print("\n" + "=" * 100)
    print("TEST 4: MEMORY SAVINGS MEASUREMENT")
    print("=" * 100)
    
    depths = [10, 50, 100, 200, 500]
    memory_results = []
    
    for d in depths:
        print(f"\nMeasuring depth {d}...")
        result = measure_memory_savings_v2(depth=d)
        memory_results.append(result)
        print(f"  Current EP:  {result.current_ep_mb:.2f} MB ({result.current_time_s:.3f}s)")
        print(f"  O(1) v2 EP:  {result.o1_ep_v2_mb:.2f} MB ({result.o1_time_s:.3f}s)")
        print(f"  Savings:     {result.savings_mb:.2f} MB ({result.savings_percent:.1f}%)")
    
    print_memory_results(memory_results)
    
    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    
    all_correctness_passed = all([grad_result.passed, settle_result.passed, training_result.passed])
    
    depth_500_result = next((r for r in memory_results if r.depth == 500), None)
    memory_target_met = depth_500_result and depth_500_result.savings_percent >= 50
    
    print(f"\nSuccess Criteria:")
    print(f"  1. Analytic gradients <1e-5 difference: {'✓' if grad_result.passed else '✗'}")
    print(f"  2. Analytic settling <1e-5 difference:  {'✓' if settle_result.passed else '✗'}")
    print(f"  3. Memory savings 50%+ at depth 500:    {'✓' if memory_target_met else '✗'}")
    
    if all_correctness_passed and memory_target_met:
        print(f"\n✅ WEEK 3-4 SUCCESS: All criteria met!")
    elif all_correctness_passed:
        print(f"\n⚠️  PARTIAL SUCCESS: Correctness verified, memory savings below target")
    else:
        print(f"\n❌ VERIFICATION FAILED: Correctness issues detected")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
