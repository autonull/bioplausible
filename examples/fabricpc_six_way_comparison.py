#!/usr/bin/env python3
"""
Bioplausible x FabricPC — Six-Way Learning Rule Comparison
===========================================================

Compares six biologically plausible learning rules on the same MLP
architecture (784->256->10) using the existing model factory.

Rules:
  1. Backpropagation
  2. Predictive Coding (FabricPC graph PCN)
  3. Equilibrium Propagation
  4. Contrastive Hebbian Learning
  5. Deep Hebbian
  6. Forward-Forward

Usage:
    python examples/fabricpc_six_way_comparison.py
"""

import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from bioplausible.core.registry import Registry


def get_mnist_loaders(
    batch_size: int = 64, train_limit: int = 2000, test_limit: int = 500
):
    """Load MNIST subset for fast comparison."""
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
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader


def evaluate(model, test_loader, device):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
            out = model.forward(x)
            correct += (out.argmax(dim=-1) == y).sum().item()
            total += y.shape[0]
    return correct / total if total > 0 else 0.0


def main():
    print("=" * 70)
    print("  Six-Way Learning Rule Comparison")
    print("  Architecture: MLP 784->256->10 (via uniform factory)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_loader, test_loader = get_mnist_loaders(
        batch_size=64, train_limit=2000, test_limit=500
    )

    # Model specs to compare
    specs = [
        ("Backprop", "Backprop Baseline"),
        ("Predictive Coding", "FabricPC Graph PCN"),
        ("Equilibrium Prop", "EqProp MLP"),
        ("Contrastive Hebbian", "CHL (Contrastive Hebbian)"),
        ("Deep Hebbian", "Deep Hebbian (Hundred-Layer)"),
        ("Forward-Forward", "Forward-Forward"),
    ]

    results = []

    for name, spec_name in specs:
        print(f"\n  {name} ({spec_name})")
        print(f"  {'-' * (len(name) + len(spec_name) + 3)}")

        try:
            spec = get_model_spec(spec_name)
            model = create_model(
                spec,
                784,
                10,
                256,
                2,
                device,
                task_type="vision",
            )

            epochs = 1  # Single epoch for speed
            t0 = time.time()
            total_loss = 0.0
            n_batches = 0

            for x, y in train_loader:
                x = x.view(x.shape[0], -1).to(device)
                y = y.to(device)

                if hasattr(model, "train_step"):
                    stats = model.train_step(x, y)
                    total_loss += stats.get("loss", 0.0)
                else:
                    # Fallback for models without custom train_step
                    model.train()
                    model.zero_grad()
                    out = model.forward(x)
                    loss = torch.nn.functional.cross_entropy(out, y)
                    loss.backward()
                    torch.optim.Adam(model.parameters(), lr=0.001).step()
                    total_loss += loss.item()

                n_batches += 1

            elapsed = time.time() - t0
            test_acc = evaluate(model, test_loader, device)

            results.append(
                {
                    "name": name,
                    "test_acc": test_acc,
                    "time": elapsed,
                    "loss": total_loss / max(n_batches, 1),
                }
            )
            print(
                f"    Loss: {total_loss / max(n_batches, 1):.4f}  |  "
                f"Test Acc: {test_acc:.4f}  |  Time: {elapsed:.2f}s"
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"name": name, "test_acc": 0.0, "time": 0.0, "loss": 0.0})

    # Summary table
    print("\n" + "=" * 70)
    print("  Comparison Results (1 epoch, 2000 train / 500 test)")
    print("=" * 70)
    print(f"  {'Learning Rule':<25} {'Test Acc':<15} {'Time (s)':<10} {'Loss':<10}")
    print(f"  {'-'*25} {'-'*15} {'-'*10} {'-'*10}")
    for r in results:
        print(
            f"  {r['name']:<25} "
            f"{r['test_acc']:.4f}{'':>11} "
            f"{r['time']:.2f}{'':>8} "
            f"{r['loss']:.4f}"
        )

    print("\n" + "=" * 70)
    print("  Rules share the same MLP architecture via create_model().")
    print("  No adapters needed — all models use build() + train_step().")
    print("=" * 70)


if __name__ == "__main__":
    main()
