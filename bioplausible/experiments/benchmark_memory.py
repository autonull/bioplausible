import gc
import os
import time

import psutil
import torch
# Disable torch compile to avoid compilation overhead noise in benchmark
import torch._dynamo
import torch.nn as nn

from bioplausible.models.looped_mlp import LoopedMLP

torch._dynamo.config.suppress_errors = True


def measure_memory_and_time(steps, gradient_method):
    input_dim = 1000
    hidden_dim = 1000
    output_dim = 10
    batch_size = 64

    device = torch.device("cpu")

    model = LoopedMLP(
        input_dim,
        hidden_dim,
        output_dim,
        max_steps=steps,
        gradient_method=gradient_method,
        use_spectral_norm=False,
    ).to(device)

    x = torch.randn(batch_size, input_dim, requires_grad=True).to(device)
    target = torch.randint(0, output_dim, (batch_size,)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Force gc
    gc.collect()

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss

    start_time = time.time()

    # Forward + Backward
    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    end_time = time.time()

    mem_end = process.memory_info().rss
    # This delta includes the graph + gradients + etc.
    mem_delta = mem_end - mem_start

    return mem_delta, end_time - start_time


def run_benchmark():
    print("Benchmarking Memory Usage: BPTT vs Equilibrium (CPU)")
    print(f"{'Method':<15} {'Steps':<10} {'Time (s)':<10} {'Mem Delta (MB)':<15}")
    print("-" * 50)

    # Warmup to load libraries
    measure_memory_and_time(5, "bptt")

    steps_list = [20, 50, 100, 200]

    for steps in steps_list:
        # BPTT
        mem_bptt, time_bptt = measure_memory_and_time(steps, "bptt")
        print(
            f"{'BPTT':<15} {steps:<10} {time_bptt:<10.4f} {mem_bptt / 1024**2:<15.2f}"
        )

        # Equilibrium
        mem_eq, time_eq = measure_memory_and_time(steps, "equilibrium")
        print(f"{'EqProp':<15} {steps:<10} {time_eq:<10.4f} {mem_eq / 1024**2:<15.2f}")


if __name__ == "__main__":
    run_benchmark()
