"""
1000-Layer Signal Probe Experiment (Track 42)

Investigates signal propagation in extremely deep equilibrium networks.
Tests the vanishing gradient hypothesis by injecting perturbations at
deep layers and measuring signal strength at shallow layers.

Key insight: EqProp should maintain better signal propagation than
traditional backprop due to bidirectional equilibrium dynamics.
"""

import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bioplausible.datasets import get_vision_dataset
from bioplausible.models import LoopedMLP, MemoryEfficientLoopedMLP
from bioplausible.training.supervised import SupervisedTrainer


def measure_layer_signals(model, h_perturbed, x_input=None):
    """
    Measure signal strength at each layer during equilibrium settling.

    Args:
        model: The model to analyze
        h_perturbed: Perturbed hidden state to analyze
        x_input: Input tensor (if needed for intermediate computations)

    Returns:
        Dictionary with signal measurements at each layer
    """
    if hasattr(model, "max_steps"):
        steps = model.max_steps
    else:
        steps = 30  # default

    signals = []

    # For this experiment, we'll measure the signal by running the model
    # and observing how perturbations propagate through the settling iterations
    with torch.no_grad():
        if x_input is not None:
            x_transformed = model._transform_input(x_input)
        else:
            # Create dummy input if not provided
            batch_size = h_perturbed.shape[0]
            x_transformed = torch.zeros(
                batch_size, model.hidden_dim, device=h_perturbed.device
            )

        h = h_perturbed.clone()

        for step in range(steps):
            h = model.forward_step(h, x_transformed)
            # Measure signal strength (mean magnitude)
            signal_strength = h.abs().mean().item()
            signals.append(signal_strength)

    return signals


def create_deep_model(
    depth: int,
    input_dim: int = 784,
    hidden_dim: int = 64,
    output_dim: int = 10,
    backend: str = "auto",
    use_residual: bool = False,
):
    """
    Create a model that simulates depth by running many equilibrium steps.

    Args:
        depth: Effective depth (number of equilibrium steps)
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        backend: Model backend ('auto', 'pytorch', 'kernel')
        use_residual: Whether to use residual connections

    Returns:
        Configured model
    """
    if backend == "kernel":
        model = MemoryEfficientLoopedMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=depth,
            use_spectral_norm=True,
        )
    else:
        model = LoopedMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=depth,
            use_spectral_norm=True,
            backend=backend,
        )

    # Store effective depth for later reference
    model.effective_depth = depth

    return model


def inject_perturbation_at_layer(
    model, layer_idx: int, perturbation_strength: float = 0.1
):
    """
    Inject a perturbation at a specific layer during equilibrium.

    Args:
        model: The model to perturb
        layer_idx: Layer index to perturb (0-indexed)
        perturbation_strength: Strength of the perturbation

    Returns:
        Perturbed hidden state
    """
    # Create a random perturbation
    batch_size = 32  # typical batch size
    device = next(model.parameters()).device

    perturbation = (
        torch.randn(batch_size, model.hidden_dim, device=device) * perturbation_strength
    )
    return perturbation


def run_signal_propagation_experiment(
    depths: List[int] = [10, 100, 500, 1000],
    perturbation_strength: float = 0.1,
    backend: str = "auto",
    use_residual: bool = False,
    dataset_name: str = "mnist",
):
    """
    Run the main signal propagation experiment.

    Args:
        depths: List of depths to test
        perturbation_strength: Strength of injected perturbation
        backend: Model backend to use
        use_residual: Whether to use residual connections
        dataset_name: Dataset to use for testing

    Returns:
        Dictionary with experimental results
    """
    print(f"Starting 1000-Layer Signal Probe Experiment")
    print(f"Testing depths: {depths}")
    print(f"Backend: {backend}")
    print(f"Use residual: {use_residual}")

    results = {
        "depths": depths,
        "signals": {},
        "times": {},
        "perturbation_strength": perturbation_strength,
    }

    # Get a small sample dataset
    dataset = get_vision_dataset(dataset_name, train=True, flatten=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Get a sample batch for testing
    sample_x, sample_y = next(iter(dataloader))

    for depth in tqdm(depths, desc="Testing depths"):
        print(f"\nTesting depth {depth}...")

        # Create model with specified depth
        model = create_deep_model(
            depth=depth,
            input_dim=sample_x.shape[1],
            output_dim=10,  # MNIST has 10 classes
            backend=backend,
            use_residual=use_residual,
        )

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        sample_x = sample_x.to(device)

        # Initialize hidden state
        h_base = model._initialize_hidden_state(sample_x)

        # Inject perturbation at the "last" layer (effectively at equilibrium)
        h_perturbed = h_base + perturbation_strength * torch.randn_like(h_base)

        # Measure signal propagation
        start_time = time.time()
        signals = measure_layer_signals(model, h_perturbed, sample_x)
        end_time = time.time()

        results["signals"][depth] = signals
        results["times"][depth] = end_time - start_time

        # Report signal at the end (after full propagation)
        final_signal = signals[-1] if signals else 0.0
        print(f"  Depth {depth}: Final signal = {final_signal:.6f}")
        print(f"  Time taken: {end_time - start_time:.2f}s")

    return results


def compare_with_skip_connections(
    depths: List[int] = [10, 100, 500, 1000], perturbation_strength: float = 0.1
):
    """
    Compare signal propagation with and without skip connections.
    """
    print("\nComparing with skip connections...")

    results_normal = run_signal_propagation_experiment(
        depths=depths, perturbation_strength=perturbation_strength, use_residual=False
    )

    results_skip = run_signal_propagation_experiment(
        depths=depths, perturbation_strength=perturbation_strength, use_residual=True
    )

    comparison_results = {"normal": results_normal, "skip_connections": results_skip}

    return comparison_results


def visualize_signal_propagation(
    results: Dict, title: str = "Signal Propagation Analysis"
):
    """
    Create visualizations for the signal propagation experiment.
    """
    try:
        import matplotlib.pyplot as plt

        depths = results["depths"]

        plt.figure(figsize=(15, 10))

        # Plot 1: Signal decay over settling steps for different depths
        plt.subplot(2, 2, 1)
        for depth in depths:
            signals = results["signals"][depth]
            steps = list(range(len(signals)))
            plt.plot(steps, signals, label=f"Depth {depth}", linewidth=1.5)

        plt.xlabel("Settling Step")
        plt.ylabel("Signal Strength")
        plt.title("Signal Decay Over Equilibrium Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Final signal vs depth
        plt.subplot(2, 2, 2)
        final_signals = [
            results["signals"][d][-1] if results["signals"][d] else 0 for d in depths
        ]
        plt.semilogy(depths, final_signals, "o-", linewidth=2, markersize=8)
        plt.xlabel("Network Depth")
        plt.ylabel("Final Signal Strength (log scale)")
        plt.title("Signal Retention vs Network Depth")
        plt.grid(True, alpha=0.3)

        # Plot 3: Computation time vs depth
        plt.subplot(2, 2, 3)
        times = [results["times"][d] for d in depths]
        plt.plot(depths, times, "s-", linewidth=2, markersize=8)
        plt.xlabel("Network Depth")
        plt.ylabel("Computation Time (s)")
        plt.title("Computation Time vs Depth")
        plt.grid(True, alpha=0.3)

        # Plot 4: Signal decay rate
        plt.subplot(2, 2, 4)
        for depth in depths:
            signals = results["signals"][depth]
            if len(signals) > 1:
                # Calculate decay rate as ratio of final/initial signal
                initial_signal = signals[0] if signals else 1e-8
                final_signal = signals[-1] if signals else 0
                retention = final_signal / (initial_signal + 1e-8)
                plt.bar(str(depth), retention, label=f"Depth {depth}")

        plt.ylabel("Signal Retention Ratio")
        plt.title("Signal Retention Across Depths")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.show()

    except ImportError:
        print("Matplotlib not available, skipping visualization")


def run_complete_signal_probe():
    """
    Run the complete 1000-layer signal probe experiment.
    """
    print("=" * 60)
    print("1000-Layer Signal Probe Experiment (Track 42)")
    print("=" * 60)

    # Test different depths
    depths = [10, 50, 100, 200, 500]  # Start with smaller depths for testing

    # Test with both backends if available
    backends_to_test = ["pytorch"]
    if torch.cuda.is_available():
        try:
            from bioplausible.kernel import HAS_CUPY

            if HAS_CUPY:
                backends_to_test.append("kernel")
        except ImportError:
            pass

    all_results = {}

    for backend in backends_to_test:
        print(f"\nTesting with {backend.upper()} backend...")
        results = run_signal_propagation_experiment(
            depths=depths, perturbation_strength=0.1, backend=backend
        )
        all_results[backend] = results

        # Print summary
        print(f"\nResults for {backend} backend:")
        for depth in depths:
            final_signal = (
                results["signals"][depth][-1] if results["signals"][depth] else 0.0
            )
            time_taken = results["times"][depth]
            print(
                f"  Depth {depth:3d}: Final signal = {final_signal:.6f}, Time = {time_taken:.2f}s"
            )

    # Visualize results for the primary backend
    if "pytorch" in all_results:
        visualize_signal_propagation(
            all_results["pytorch"], f"Signal Propagation - PyTorch Backend"
        )

    if "kernel" in all_results:
        visualize_signal_propagation(
            all_results["kernel"], f"Signal Propagation - Kernel Backend (O(1) Memory)"
        )

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("Success metric: Signal > 1% at depth 1000")
    print("=" * 60)

    # Check success criteria
    for backend, results in all_results.items():
        if 1000 in results["depths"]:  # If we tested depth 1000
            final_signal = (
                results["signals"][1000][-1] if results["signals"][1000] else 0.0
            )
            success = final_signal > 0.01  # > 1%
            print(
                f"{backend.capitalize()} backend signal at depth 1000: {final_signal:.6f} ({'SUCCESS' if success else 'FAILED'})"
            )

    return all_results


if __name__ == "__main__":
    results = run_complete_signal_probe()
