#!/usr/bin/env python3
"""
Profile the EP settling loop to identify bottlenecks.

Run: python examples/profile_settling.py
"""

import torch
import torch.nn as nn
import time
from torch.profiler import profile, ProfilerActivity

from mep import smep
from mep.optimizers.settling import Settler
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector


def profile_settling():
    """Profile the settling loop directly."""
    print("=" * 60)
    print("Profiling EP Settling Loop")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    
    # Setup
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    energy_fn = EnergyFunction(loss_type="cross_entropy")
    settler = Settler(steps=15, lr=0.1, adaptive=True, tol=1e-3, patience=3)
    
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    # Warmup
    for _ in range(3):
        _ = settler.settle(model, x, y, beta=0.3, energy_fn=energy_fn, structure=structure)
    
    # Profile
    print("\nProfiling 10 settling iterations...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            _ = settler.settle(model, x, y, beta=0.3, energy_fn=energy_fn, structure=structure)
    
    print("\n" + "=" * 60)
    print("Top Operations by CPU Time")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    
    if device == "cuda":
        print("\n" + "=" * 60)
        print("Top Operations by CUDA Time")
        print("=" * 60)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def profile_full_training():
    """Profile full EP training step."""
    print("\n" + "=" * 60)
    print("Profiling Full EP Training Step")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    
    optimizer = smep(
        model.parameters(),
        model=model,
        mode="ep",
        settle_steps=15,
        settle_lr=0.1,
        beta=0.3,
    )
    
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    # Warmup
    for _ in range(3):
        optimizer.step(x=x, target=y)
        optimizer.zero_grad()
    
    # Profile
    print("Profiling 20 training steps...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(20):
            optimizer.step(x=x, target=y)
            optimizer.zero_grad()
    
    print("\n" + "=" * 60)
    print("Top Operations by Total Time")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


def benchmark_adaptive_vs_fixed():
    """Compare adaptive vs fixed settling."""
    print("\n" + "=" * 60)
    print("Benchmark: Adaptive vs Fixed Settling")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    
    inspector = ModelInspector()
    structure = inspector.inspect(model)
    energy_fn = EnergyFunction(loss_type="cross_entropy")
    
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    # Fixed settling (no early stop)
    settler_fixed = Settler(steps=15, lr=0.1, adaptive=False)
    
    start = time.time()
    for _ in range(50):
        _ = settler_fixed.settle(model, x, y, beta=0.3, energy_fn=energy_fn, structure=structure)
    fixed_time = time.time() - start
    
    # Adaptive settling
    settler_adaptive = Settler(steps=15, lr=0.1, adaptive=True, tol=1e-3, patience=3)
    
    start = time.time()
    for _ in range(50):
        _ = settler_adaptive.settle(model, x, y, beta=0.3, energy_fn=energy_fn, structure=structure)
    adaptive_time = time.time() - start
    
    speedup = fixed_time / adaptive_time if adaptive_time > 0 else float('inf')
    
    print(f"Fixed settling (15 steps):    {fixed_time*1000:.1f}ms for 50 iterations")
    print(f"Adaptive settling (early stop): {adaptive_time*1000:.1f}ms for 50 iterations")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    profile_settling()
    profile_full_training()
    benchmark_adaptive_vs_fixed()
