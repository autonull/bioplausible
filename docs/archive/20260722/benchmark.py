"""
Benchmark Script for Bioplausible Models
"""

import argparse
import time
from typing import Any

import torch
from bioplausible.models.conv_eqprop import ConvEqProp
from bioplausible.models.looped_mlp import LoopedMLP
from bioplausible.models.modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from bioplausible.models.transformer_eqprop import TransformerEqProp

from bioplausible.utils import profile_model


def run_benchmark(model_name: str, device: str = "cpu") -> dict[str, Any]:
    print(f"Benchmarking {model_name} on {device}...")

    if model_name.startswith("looped_mlp"):
        # Check for gradient method
        grad_method = "equilibrium" if "deq" in model_name else "bptt"
        model = LoopedMLP(784, 512, 10, max_steps=30)
        # Assuming EqPropModel now accepts gradient_method in init or property
        if hasattr(model, "gradient_method"):
            model.gradient_method = grad_method
        input_shape = (32, 784)
    elif model_name == "conv_eqprop":
        model = ConvEqProp(3, 64, 10, max_steps=20)
        input_shape = (32, 3, 32, 32)
    elif model_name == "modern_conv":
        model = ModernConvEqProp(eq_steps=15)
        input_shape = (32, 3, 32, 32)
    elif model_name == "simple_conv":
        model = SimpleConvEqProp(eq_steps=20)
        input_shape = (32, 3, 32, 32)
    elif model_name == "transformer":
        model = TransformerEqProp(1000, 256, 10, num_layers=2, max_steps=10)
        input_shape = (32, 64)  # batch, seq_len
        # Transformer input needs integer tokens
        model = model.to(device)
        x = torch.randint(0, 1000, input_shape, device=device)

        # Custom profile for transformer (due to int input)
        model.eval()
        times = []
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(x)

            for _ in range(20):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        print(f"  Avg: {avg:.2f} ms +/- {std:.2f}")
        return {"model": model_name, "avg_ms": avg, "std_ms": std}

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Standard profile for continuous input models
    stats = profile_model(model, input_shape, device=device, runs=20)
    print(f"  Avg: {stats['avg_ms']:.2f} ms +/- {stats['std_ms']:.2f}")
    return {**stats, "model": model_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    models = [
        "looped_mlp",
        "looped_mlp_deq",
        "conv_eqprop",
        "modern_conv",
        "simple_conv",
        "transformer",
    ]

    results = []
    for m in models:
        try:
            res = run_benchmark(m, args.device)
            results.append(res)
        except Exception as e:
            print(f"Failed to benchmark {m}: {e}")

    print("\nSummary:")
    print(f"{'Model':<15} | {'Avg (ms)':<10} | {'Std (ms)':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['model']:<15} | {r['avg_ms']:<10.2f} | {r['std_ms']:<10.2f}")


if __name__ == "__main__":
    main()
