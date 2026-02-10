#!/usr/bin/env python3
"""
Track 36: Energy-Based OOD Detection

Hypothesis: Energy-based confidence outperforms softmax for OOD detection.

Uses energy score = -E / (settling_time + 1) where:
- Lower energy = higher confidence
- Faster settling = higher confidence
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bioplausible.models import LoopedMLP


def compute_energy_score(model, x, max_steps=30, epsilon=1e-5):
    """
    Compute energy-based confidence score.

    Returns:
        dict with 'energy', 'settling_time', 'score'
    """
    model.eval()
    with torch.no_grad():
        # Build trajectory
        batch_size = x.size(0)
        h = torch.zeros(batch_size, model.hidden_dim, device=x.device)
        x_proj = model.W_in(x.view(x.size(0), -1))

        h_trajectory = [h.clone()]

        for step in range(max_steps):
            h_new = torch.tanh(x_proj + model.W_rec(h))
            h_trajectory.append(h_new)

            # Check convergence
            delta = (h_new - h).norm(dim=1).mean().item()
            h = h_new

            if delta < epsilon:
                settling_time = step + 1
                break
        else:
            settling_time = max_steps

        # Compute energy (final reconstruction error)
        out = model.W_out(h)
        # Energy = magnitude of hidden state (proxy for instability)
        energy = h.norm(dim=1).mean().item()

        # Combined score: lower energy and faster settling = higher confidence
        score = -energy / (settling_time + 1)

    return {
        "energy": energy,
        "settling_time": settling_time,
        "score": score,
    }


def compute_softmax_score(model, x):
    """Compute softmax-based confidence (max probability)."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
    return confidence


def load_dataset(name, batch_size=128):
    """Load a dataset."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Simple normalization
        ]
    )

    if name == "CIFAR10":
        dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif name == "SVHN":
        dataset = datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )
    elif name == "CIFAR100":
        dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Convert to grayscale by averaging channels
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def run_ood_experiment(model, device="cuda"):
    """Run OOD detection experiment."""
    print("=" * 60)
    print("TRACK 36: Energy-Based OOD Detection")
    print("=" * 60)
    print()

    # In-distribution: CIFAR-10
    print("Loading CIFAR-10 (in-distribution)...")
    id_loader = load_dataset("CIFAR10", batch_size=100)

    # OOD datasets
    print("Loading OOD datasets...")
    ood_loaders = {
        "SVHN": load_dataset("SVHN", batch_size=100),
        "CIFAR100": load_dataset("CIFAR100", batch_size=100),
    }

    # Also test on Gaussian noise
    print("Generating Gaussian noise...")
    gaussian_noise = torch.randn(1000, 3, 32, 32)

    model = model.to(device)

    # Collect scores
    print("\nComputing scores...")

    # ID scores
    print("  [1/4] In-distribution (CIFAR-10)...")
    id_energy_scores = []
    id_softmax_scores = []

    for x, _ in id_loader:
        x = x.to(device)

        # Energy score (batch-wise)
        result = compute_energy_score(model, x)
        id_energy_scores.append(result["score"])

        # Softmax score
        sm_score = compute_softmax_score(model, x)
        id_softmax_scores.append(sm_score)

        if len(id_energy_scores) >= 10:  # Limit for speed
            break

    # OOD scores
    ood_results = {}

    for name, loader in ood_loaders.items():
        print(f"  [OOD] {name}...")
        ood_energy_scores = []
        ood_softmax_scores = []

        for x, _ in loader:
            x = x.to(device)

            result = compute_energy_score(model, x)
            ood_energy_scores.append(result["score"])

            sm_score = compute_softmax_score(model, x)
            ood_softmax_scores.append(sm_score)

            if len(ood_energy_scores) >= 10:
                break

        ood_results[name] = {
            "energy_scores": ood_energy_scores,
            "softmax_scores": ood_softmax_scores,
        }

    # Gaussian noise
    print(f"  [OOD] Gaussian Noise...")
    noise_energy_scores = []
    noise_softmax_scores = []

    for i in range(0, len(gaussian_noise), 100):
        x = gaussian_noise[i : i + 100].to(device)
        result = compute_energy_score(model, x)
        noise_energy_scores.append(result["score"])

        sm_score = compute_softmax_score(model, x)
        noise_softmax_scores.append(sm_score)

    ood_results["Gaussian"] = {
        "energy_scores": noise_energy_scores,
        "softmax_scores": noise_softmax_scores,
    }

    # Compute AUROC
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    auroc_results = {}

    for name, scores in ood_results.items():
        # Energy AUROC
        labels = [0] * len(id_energy_scores) + [1] * len(scores["energy_scores"])
        predictions = id_energy_scores + scores["energy_scores"]
        energy_auroc = roc_auc_score(labels, predictions)

        # Softmax AUROC
        predictions_sm = id_softmax_scores + scores["softmax_scores"]
        softmax_auroc = roc_auc_score(labels, predictions_sm)

        print(f"\n{name}:")
        print(f"  Energy AUROC:  {energy_auroc:.3f}")
        print(f"  Softmax AUROC: {softmax_auroc:.3f}")
        print(f"  Improvement:   {(energy_auroc - softmax_auroc)*100:+.1f}%")

        auroc_results[name] = {
            "energy_auroc": energy_auroc,
            "softmax_auroc": softmax_auroc,
        }

    # Check success
    energy_aurocs = [r["energy_auroc"] for r in auroc_results.values()]
    avg_auroc = np.mean(energy_aurocs)

    print(f"\nAverage Energy AUROC: {avg_auroc:.3f}")
    print(f"Target: ≥ 0.85")

    if avg_auroc >= 0.85:
        print("✅ TARGET ACHIEVED")
    elif avg_auroc >= 0.80:
        print("⚠️ Close to target")
    else:
        print("❌ Did not achieve target")

    return auroc_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a simple trained model
    print("Creating model...")
    model = LoopedMLP(3072, 256, 10, use_spectral_norm=True, max_steps=30)

    # For demo, just use random weights
    # In practice, would load a trained checkpoint
    print("⚠️ Using untrained model for demo (results will be random)")
    print("   Load a trained checkpoint for real results")
    print()

    results = run_ood_experiment(model, device=device)

    # Save results
    save_dir = Path("results/track_36")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "ood_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
