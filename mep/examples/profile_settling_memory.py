#!/usr/bin/env python3
"""
O(1) Memory: Settling Phase Profiling

Phase 2: Week 3-4 - Isolate settling memory from contrast memory

Key insight: We need to measure settling memory separately from contrast memory.
The settling phase should be O(1), while contrast still uses O(depth) for parameter gradients.

Run: python examples/profile_settling_memory.py
"""

import torch
import torch.nn as nn
import gc
import time
from typing import List, Dict, Any

from mep import smep, Settler, EnergyFunction, ModelInspector
from mep.optimizers import (
    settle_manual_o1,
    analytic_state_gradients,
    energy_from_states_minimal,
)


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
    return DeepMLP(input_dim, hidden_dim, num_layers, output_dim).to(device)


def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.1)


def get_peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_stats().get('allocated_bytes.all.peak', 0) / 1e6
    return 0.0


def measure_settling_memory(
    depth: int,
    input_dim: int = 784,
    hidden_dim: int = 128,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Measure memory used ONLY during settling (free phase).
    
    This isolates the settling memory from contrast memory.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x = torch.randn(batch_size, input_dim, device=device)
    model = make_model(input_dim, hidden_dim, depth, 10, device)
    
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    # Current EP settling (with autograd)
    settler = Settler(steps=30, lr=0.15)
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    states_current = settler.settle(model, x, None, beta=0.0, energy_fn=energy_fn, structure=structure)
    
    settling_mem_current = get_peak_memory_mb()
    
    del states_current, model
    reset_memory()
    
    # O(1) settling (analytic gradients)
    model_o1 = make_model(input_dim, hidden_dim, depth, 10, device)
    inspector_o1 = ModelInspector()  # Fresh inspector
    structure_o1 = inspector_o1.inspect(model_o1)  # Fresh structure
    
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    states_o1 = settle_manual_o1(model_o1, x, None, beta=0.0, structure=structure_o1, steps=30, lr=0.15)
    
    settling_mem_o1 = get_peak_memory_mb()
    
    del states_o1, model_o1
    
    return {
        'depth': depth,
        'current_settling_mb': settling_mem_current,
        'o1_settling_mb': settling_mem_o1,
        'settling_savings_mb': settling_mem_current - settling_mem_o1,
        'settling_savings_percent': (settling_mem_current - settling_mem_o1) / settling_mem_current * 100 if settling_mem_current > 0 else 0,
    }


def measure_contrast_memory(
    depth: int,
    input_dim: int = 784,
    hidden_dim: int = 128,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Measure memory used ONLY during contrast step.
    
    This measures the memory for computing parameter gradients from settled states.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x = torch.randn(batch_size, input_dim, device=device)
    
    # Current EP contrast (full graph)
    model = make_model(input_dim, hidden_dim, depth, 10, device)
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    
    settler = Settler(steps=30, lr=0.15)
    energy_fn = EnergyFunction(loss_type='cross_entropy')
    
    states_free = settler.settle(model, x, None, beta=0.0, energy_fn=energy_fn, structure=structure)
    states_nudged = settler.settle(model, x, None, beta=0.5, energy_fn=energy_fn, structure=structure)
    
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    E_free_current = energy_fn(model, x, states_free, structure, None, 0.0)
    E_nudged_current = energy_fn(model, x, states_nudged, structure, None, 0.5)
    contrast_current = (E_nudged_current - E_free_current) / 0.5
    
    params = list(model.parameters())
    grads_current = torch.autograd.grad(contrast_current, params, retain_graph=False)
    
    contrast_mem_current = get_peak_memory_mb()
    
    del grads_current, E_free_current, E_nudged_current, contrast_current, states_free, states_nudged, model
    reset_memory()
    
    # O(1) contrast (gradient checkpointing)
    model_o1 = make_model(input_dim, hidden_dim, depth, 10, device)
    inspector_o1 = ModelInspector()
    structure_o1 = inspector_o1.inspect(model_o1)
    
    states_free_o1 = settler.settle(model_o1, x, None, beta=0.0, energy_fn=energy_fn, structure=structure_o1)
    states_nudged_o1 = settler.settle(model_o1, x, None, beta=0.5, energy_fn=energy_fn, structure=structure_o1)
    
    reset_memory()
    torch.cuda.reset_peak_memory_stats()
    
    E_free_o1 = energy_from_states_minimal(model_o1, x, states_free_o1, structure_o1, None, 0.0, loss_type='cross_entropy')
    E_nudged_o1 = energy_from_states_minimal(model_o1, x, states_nudged_o1, structure_o1, None, 0.5, loss_type='cross_entropy')
    contrast_o1 = (E_nudged_o1 - E_free_o1) / 0.5
    
    params_o1 = list(model_o1.parameters())
    grads_o1 = torch.autograd.grad(contrast_o1, params_o1, retain_graph=False)
    
    contrast_mem_o1 = get_peak_memory_mb()
    
    del grads_o1, model_o1
    
    return {
        'depth': depth,
        'current_contrast_mb': contrast_mem_current,
        'o1_contrast_mb': contrast_mem_o1,
        'contrast_savings_mb': contrast_mem_current - contrast_mem_o1,
        'contrast_savings_percent': (contrast_mem_current - contrast_mem_o1) / contrast_mem_current * 100 if contrast_mem_current > 0 else 0,
    }


def measure_full_step_memory(
    depth: int,
    input_dim: int = 784,
    hidden_dim: int = 128,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Measure memory for complete training step (settling + contrast).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    # Current EP
    model_current = make_model(input_dim, hidden_dim, depth, 10, device)
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
    torch.cuda.reset_peak_memory_stats()
    
    optimizer_current.step(x=x, target=y)
    
    mem_current = get_peak_memory_mb()
    
    del model_current, optimizer_current
    reset_memory()
    
    # O(1) EP v2
    model_o1 = make_model(input_dim, hidden_dim, depth, 10, device)
    from mep.optimizers import O1MemoryEPv2
    
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
    torch.cuda.reset_peak_memory_stats()
    
    optimizer_o1.step(x=x, target=y)
    
    mem_o1 = get_peak_memory_mb()
    
    del model_o1, optimizer_o1
    
    return {
        'depth': depth,
        'current_full_mb': mem_current,
        'o1_full_mb': mem_o1,
        'full_savings_mb': mem_current - mem_o1,
        'full_savings_percent': (mem_current - mem_o1) / mem_current * 100 if mem_current > 0 else 0,
    }


def print_table(title: str, results: List[Dict[str, float]], columns: List[str]):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    
    header = f"{'Depth':<8}"
    for col in columns:
        header += f" {col:<18}"
    print(header)
    print("-" * 100)
    
    for r in results:
        row = f"{r['depth']:<8}"
        for col in columns:
            key = col.lower().replace(' ', '_').replace('(', '').replace(')', '')
            val = f"{r.get(key, 0):.2f}"
            row += f" {val:<18}"
        print(row)
    
    print("=" * 100)


def main():
    print("=" * 100)
    print("O(1) MEMORY: PHASE BREAKDOWN")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available.\n")
    
    depths = [10, 50, 100, 200, 500]
    
    # Measure settling memory
    print("\nMeasuring settling phase memory...")
    settling_results = []
    for d in depths:
        result = measure_settling_memory(depth=d)
        settling_results.append(result)
        print(f"  Depth {d}: Current={result['current_settling_mb']:.2f}MB, O(1)={result['o1_settling_mb']:.2f}MB, Savings={result['settling_savings_percent']:.1f}%")
    
    # Measure contrast memory
    print("\nMeasuring contrast phase memory...")
    contrast_results = []
    for d in depths:
        result = measure_contrast_memory(depth=d)
        contrast_results.append(result)
        print(f"  Depth {d}: Current={result['current_contrast_mb']:.2f}MB, O(1)={result['o1_contrast_mb']:.2f}MB, Savings={result['contrast_savings_percent']:.1f}%")
    
    # Measure full step memory
    print("\nMeasuring full step memory...")
    full_results = []
    for d in depths:
        result = measure_full_step_memory(depth=d)
        full_results.append(result)
        print(f"  Depth {d}: Current={result['current_full_mb']:.2f}MB, O(1)={result['o1_full_mb']:.2f}MB, Savings={result['full_savings_percent']:.1f}%")
    
    # Print tables
    print_table(
        "SETTLING PHASE MEMORY",
        settling_results,
        ["Current (MB)", "O(1) (MB)", "Savings (MB)", "Savings (%)"]
    )
    
    print_table(
        "CONTRAST PHASE MEMORY",
        contrast_results,
        ["Current (MB)", "O(1) (MB)", "Savings (MB)", "Savings (%)"]
    )
    
    print_table(
        "FULL STEP MEMORY",
        full_results,
        ["Current (MB)", "O(1) (MB)", "Savings (MB)", "Savings (%)"]
    )
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if settling_results:
        settling_savings = [r['settling_savings_percent'] for r in settling_results]
        print(f"\nSettling Phase:")
        print(f"  Average savings: {sum(settling_savings)/len(settling_savings):.1f}%")
        print(f"  Max savings at depth {max(settling_results, key=lambda x: x['depth'])['depth']}: {settling_results[-1]['settling_savings_percent']:.1f}%")
    
    if contrast_results:
        contrast_savings = [r['contrast_savings_percent'] for r in contrast_results]
        print(f"\nContrast Phase:")
        print(f"  Average savings: {sum(contrast_savings)/len(contrast_savings):.1f}%")
        print(f"  Max savings at depth {max(contrast_results, key=lambda x: x['depth'])['depth']}: {contrast_results[-1]['contrast_savings_percent']:.1f}%")
    
    if full_results:
        full_savings = [r['full_savings_percent'] for r in full_results]
        print(f"\nFull Step:")
        print(f"  Average savings: {sum(full_savings)/len(full_savings):.1f}%")
        print(f"  Max savings at depth {max(full_results, key=lambda x: x['depth'])['depth']}: {full_results[-1]['full_savings_percent']:.1f}%")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
