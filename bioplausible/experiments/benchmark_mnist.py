#!/usr/bin/env python3
"""
Benchmark Script: MNIST Comparison

Compares three bio-plausible approaches on MNIST:
1. LoopedMLP (Standard BPTT on Equilibrium Model)
2. StandardEqProp (Equilibrium Propagation with Contrastive Updates)
3. StandardFA (Feedback Alignment)

Reports training time, final accuracy, and convergence speed.
"""

import time

import torch
import torch.nn as nn

from bioplausible.core import EqPropTrainer
from bioplausible.datasets import create_data_loaders
from bioplausible.models import LoopedMLP, StandardEqProp, StandardFA
from bioplausible.utils import seed_everything


def run_benchmark(model_cls, name, epochs=3, **model_kwargs):
    print(f"\n--- Benchmarking {name} ---")
    seed_everything(42)

    # Create Data
    train_loader, test_loader = create_data_loaders(
        "mnist", batch_size=128, flatten=True
    )
    input_dim = 784
    output_dim = 10

    # Create Model
    # Some models take config, some take kwargs.
    # StandardEqProp/FA use BioModel which accepts kwargs.
    # LoopedMLP uses NEBCBase which accepts kwargs.

    if name == "LoopedMLP (BPTT)":
        # LoopedMLP needs hidden_dim, not hidden_dims list usually, but let's check init.
        # It takes input_dim, hidden_dim, output_dim.
        model = model_cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=model_kwargs.get("hidden_dim", 256),
        )
    else:
        # StandardEqProp/FA
        model = model_cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[model_kwargs.get("hidden_dim", 256)],
            **model_kwargs,
        )

    print(
        f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Create Trainer
    # use_kernel=False for all to ensure fair Python comparison (EqPropKernel is specialized)
    # Check if GPU available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = EqPropTrainer(
        model,
        optimizer="adam",
        lr=0.001,
        use_kernel=False,
        use_compile=False,
        device=device,
    )

    # Train
    start_time = time.time()
    history = trainer.fit(
        train_loader, epochs=epochs, val_loader=test_loader, log_interval=100
    )
    end_time = time.time()

    duration = end_time - start_time
    final_acc = history["val_acc"][-1] if history["val_acc"] else 0.0

    print(f"Result: {final_acc:.2%} Accuracy in {duration:.2f}s")
    return {
        "name": name,
        "accuracy": final_acc,
        "duration": duration,
        "history": history,
    }


def main():
    results = []

    # 1. Looped MLP (BPTT)
    # This serves as the "Backprop" baseline on the recurrent architecture
    results.append(run_benchmark(LoopedMLP, "LoopedMLP (BPTT)", hidden_dim=256))

    # 2. Feedback Alignment
    results.append(run_benchmark(StandardFA, "Feedback Alignment", hidden_dim=256))

    # 3. Equilibrium Propagation
    # Note: EqProp often requires smaller LR or careful tuning.
    results.append(
        run_benchmark(
            StandardEqProp,
            "Equilibrium Propagation",
            hidden_dim=256,
            beta=0.5,
            equilibrium_steps=20,
        )
    )

    print("\n\n=== BENCHMARK SUMMARY ===")
    print(f"{'Model':<25} | {'Acc':<8} | {'Time':<8} | {'Speed (s/ep)':<12}")
    print("-" * 60)
    for res in results:
        epochs = len(res["history"]["train_loss"])
        print(
            f"{res['name']:<25} | {res['accuracy']:.2%}   | {res['duration']:.2f}s   | {res['duration']/epochs:.2f}"
        )


if __name__ == "__main__":
    main()
