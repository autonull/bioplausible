#!/usr/bin/env python3
"""
Bioplausible x FabricPC Integration Demo — MNIST Bridge
========================================================

Graph API and PC training mode inspired by FabricPC (Behrend et al.)
https://github.com/trueagi-io/FabricPC

Defines a 784->256->10 MLP graph once, then trains with both
backpropagation and predictive coding, reporting a comparison table.

Usage:
    python examples/fabricpc_mnist_bridge.py
"""

import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from bioplausible.graph import (
    Edge,
    InferenceSGD,
    Linear,
    ReLU,
    TaskMap,
    graph,
    initialize_params,
    train_backprop,
    train_pcn,
)


def get_mnist_loaders(
    batch_size: int = 64, train_limit: int = 0, test_limit: int = 0
):
    """Load MNIST with optional subsetting for fast iteration."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        "../data", train=False, download=True, transform=transform
    )
    if train_limit > 0:
        train_set = Subset(train_set, range(min(train_limit, len(train_set))))
    if test_limit > 0:
        test_set = Subset(test_set, range(min(test_limit, len(test_set))))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_mlp_graph(device: torch.device):
    """Define a single 784->256->10 MLP graph."""
    inp = Linear(shape=(784, 256), name="input")
    act = ReLU(name="hidden")
    out = Linear(shape=(256, 10), name="output")
    structure = graph(
        nodes=[inp, act, out],
        edges=[
            Edge(source=inp, target=act.slot("input")),
            Edge(source=act, target=out.slot("input")),
        ],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
    return structure


def main():
    print("=" * 65)
    print("  Bioplausible x FabricPC — MNIST Bridge Demo")
    print("  Same graph, two training modes: Backprop vs PC")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_loader, test_loader = get_mnist_loaders(
        batch_size=64, train_limit=0, test_limit=0
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Build graph once, initialize params separately for each run
    print("\nBuilding 784->256->10 graph...")
    structure = build_mlp_graph(device)

    params_bp = initialize_params(structure)
    params_pc = initialize_params(structure)

    for node_name in params_bp:
        for param_name in params_bp[node_name]:
            params_bp[node_name][param_name] = params_bp[node_name][
                param_name
            ].to(device)
            params_pc[node_name][param_name] = params_pc[node_name][
                param_name
            ].to(device)

    print(f"  Nodes: {len(structure.nodes)}, Edges: {len(structure.edges)}")

    # ---- Backprop ----
    print("\nTraining: Backpropagation")
    print("-" * 65)
    t0 = time.time()
    results_bp = train_backprop(
        structure,
        params_bp,
        train_loader,
        test_loader=test_loader,
        epochs=5,
        lr=0.001,
        device=device,
    )
    bp_time = time.time() - t0
    print(
        f"  Train Acc: {results_bp['train_acc']:.4f}  |  "
        f"Test Acc: {results_bp['test_acc']:.4f}  |  "
        f"Time: {bp_time:.2f}s"
    )

    # ---- PC ----
    print("\nTraining: Predictive Coding")
    print("-" * 65)
    t0 = time.time()
    results_pc = train_pcn(
        structure,
        params_pc,
        train_loader,
        test_loader=test_loader,
        epochs=5,
        lr=0.001,
        device=device,
        infer_steps=20,
        eta_infer=0.05,
    )
    pc_time = time.time() - t0
    print(
        f"  Train Acc: {results_pc['train_acc']:.4f}  |  "
        f"Test Acc: {results_pc['test_acc']:.4f}  |  "
        f"Time: {pc_time:.2f}s"
    )

    # ---- Comparison Table ----
    print("\n" + "=" * 65)
    print("  Comparison Table")
    print("=" * 65)
    header = f"  {'Learning Rule':<25} {'Test Acc':<15} {'Time (s)':<10}"
    sep = f"  {'-'*25} {'-'*15} {'-'*10}"
    print(header)
    print(sep)
    print(
        f"  {'Backpropagation':<25} "
        f"{results_bp['test_acc']:.4f}{'':>11} "
        f"{bp_time:.2f}"
    )
    print(
        f"  {'Predictive Coding':<25} "
        f"{results_pc['test_acc']:.4f}{'':>11} "
        f"{pc_time:.2f}"
    )

    diff = abs(results_bp["test_acc"] - results_pc["test_acc"])
    if results_bp["test_acc"] > results_pc["test_acc"]:
        print(f"\n  Backprop leads by {diff:.4f} accuracy")
    else:
        print(f"\n  PC leads by {diff:.4f} accuracy")

    print("\n" + "=" * 65)
    print("  Demo Complete!")
    print("  Graph API and PC mode adapted from FabricPC")
    print("  https://github.com/trueagi-io/FabricPC")
    print("=" * 65)


if __name__ == "__main__":
    main()
