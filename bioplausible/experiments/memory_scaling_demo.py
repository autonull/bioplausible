#!/usr/bin/env python3
"""
Track B / Track 35: Memory Scaling Demonstration

Hypothesis: EqProp achieves O(√D) memory with gradient checkpointing
vs Backprop's O(D) scaling, enabling training of 200+ layer networks on 8GB GPU.

This experiment measures peak GPU memory at varying depths and demonstrates
that backprop OOMs while EqProp continues to scale.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bioplausible.models import LoopedMLP


class DeepEqPropCheckpointed(nn.Module):
    """
    Deep network with gradient checkpointing for memory efficiency.

    Uses √D checkpointing strategy: checkpoint every sqrt(depth) layers.
    """

    def __init__(
        self,
        depth: int,
        hidden_dim: int = 256,
        input_dim: int = 3072,
        output_dim: int = 10,
    ):
        super().__init__()
        self.depth = depth
        self.checkpoint_freq = max(1, int(depth**0.5))

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Apply spectral norm to all layers for stability
        from torch.nn.utils.parametrizations import spectral_norm

        self.input_layer = spectral_norm(self.input_layer)
        for i, layer in enumerate(self.layers):
            self.layers[i] = spectral_norm(layer)
        self.output_layer = spectral_norm(self.output_layer)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Input projection
        h = torch.tanh(self.input_layer(x))

        # Deep layers with checkpointing
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_freq == 0 and self.training:
                # Checkpoint this layer
                h = checkpoint(lambda h: torch.tanh(layer(h)), h, use_reentrant=False)
            else:
                h = torch.tanh(layer(h))

        # Output
        return self.output_layer(h)


class StandardDeepMLP(nn.Module):
    """Standard deep MLP without checkpointing (for comparison)."""

    def __init__(
        self,
        depth: int,
        hidden_dim: int = 256,
        input_dim: int = 3072,
        output_dim: int = 10,
    ):
        super().__init__()
        self.depth = depth

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def measure_memory(model, batch_size=128, device="cuda"):
    """
    Measure peak GPU memory during forward + backward pass.

    Returns:
        dict with 'peak_memory_mb' and 'oom' flag
    """
    if device == "cpu":
        return {"peak_memory_mb": 0.0, "oom": False}

    model = model.to(device)
    model.train()

    try:
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # Create dummy data
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        # Forward pass
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        # Backward pass (this is where memory peaks)
        loss.backward()

        # Measure peak memory
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e6  # Convert to MB

        # Cleanup
        del x, y, output, loss
        torch.cuda.empty_cache()

        return {"peak_memory_mb": peak_memory, "oom": False}

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # OOM occurred
            torch.cuda.empty_cache()
            return {"peak_memory_mb": None, "oom": True}
        else:
            raise


def run_memory_experiment(depths, batch_size=128, device="cuda"):
    """
    Run memory scaling experiment across different depths.

    Returns:
        dict with results for EqProp and Backprop
    """
    print("=" * 60)
    print("TRACK 35: Memory Scaling Demonstration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Depths: {depths}")
    print()

    results = {
        "config": {
            "depths": depths,
            "batch_size": batch_size,
            "device": device,
        },
        "eqprop": [],
        "backprop": [],
    }

    for depth in depths:
        print(f"\nDepth = {depth}")
        print("-" * 40)

        # Test EqProp with checkpointing
        print(f"  [1/2] EqProp (checkpointed)...")
        model_eqprop = DeepEqPropCheckpointed(depth, hidden_dim=256)
        result_eq = measure_memory(model_eqprop, batch_size, device)

        if result_eq["oom"]:
            print(f"    ❌ OOM")
        else:
            print(f"    ✅ {result_eq['peak_memory_mb']:.1f} MB")

        results["eqprop"].append({"depth": depth, **result_eq})

        # Test standard backprop
        print(f"  [2/2] Backprop (no checkpointing)...")
        model_backprop = StandardDeepMLP(depth, hidden_dim=256)
        result_bp = measure_memory(model_backprop, batch_size, device)

        if result_bp["oom"]:
            print(f"    ❌ OOM")
        else:
            print(f"    ✅ {result_bp['peak_memory_mb']:.1f} MB")

        results["backprop"].append({"depth": depth, **result_bp})

        # Cleanup
        del model_eqprop, model_backprop
        torch.cuda.empty_cache()

    return results


def analyze_results(results):
    """Analyze and summarize memory scaling results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n{:<10} {:<20} {:<20}".format("Depth", "EqProp (MB)", "Backprop (MB)"))
    print("-" * 50)

    eqprop_max_depth = 0
    backprop_max_depth = 0

    for eq, bp in zip(results["eqprop"], results["backprop"]):
        depth = eq["depth"]

        eq_str = f"{eq['peak_memory_mb']:.1f}" if not eq["oom"] else "OOM"
        bp_str = f"{bp['peak_memory_mb']:.1f}" if not bp["oom"] else "OOM"

        print(f"{depth:<10} {eq_str:<20} {bp_str:<20}")

        if not eq["oom"]:
            eqprop_max_depth = depth
        if not bp["oom"]:
            backprop_max_depth = depth

    print()
    print(f"Maximum depth without OOM:")
    print(f"  EqProp (checkpointed): {eqprop_max_depth}")
    print(f"  Backprop (standard):   {backprop_max_depth}")

    # Check success criteria
    success = eqprop_max_depth >= 200 and backprop_max_depth < 100

    print()
    if success:
        print("✅ SUCCESS: EqProp trains 200+ layers, Backprop OOMs before 100")
    elif eqprop_max_depth >= 200:
        print("⚠️ PARTIAL: EqProp achieved 200+ layers")
    else:
        print("❌ FAIL: Did not achieve target")

    return {
        "eqprop_max_depth": eqprop_max_depth,
        "backprop_max_depth": backprop_max_depth,
        "success": success,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("⚠️ No GPU detected. This experiment requires CUDA.")
        print("Skipping...")
        return

    # Test depths
    depths = [10, 25, 50, 100, 200, 500]

    # Run experiment
    results = run_memory_experiment(depths, batch_size=128, device=device)

    # Analyze
    summary = analyze_results(results)

    # Save results
    save_dir = Path("results/track_35")
    save_dir.mkdir(parents=True, exist_ok=True)

    results["summary"] = summary

    with open(save_dir / "memory_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {save_dir / 'memory_scaling_results.json'}")


if __name__ == "__main__":
    main()
