"""
Discovery Demo: Analyzing Model Dynamics

Demonstrates how to use the DynamicsAnalyzer to "open up" the model
and inspect its convergence to equilibrium, a key property of Bio-Plausible models.
"""

import numpy as np
import torch

from bioplausible.analysis import DynamicsAnalyzer
from bioplausible.datasets import get_vision_dataset
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec


def main():
    print("Initializing Model for Analysis...")

    # Create a model manually
    spec = get_model_spec("EqProp MLP")

    # Using small dimensions for demonstration
    model = create_model(
        spec=spec,
        input_dim=64,  # Digits dataset is 8x8 = 64
        output_dim=10,
        hidden_dim=256,
        device="cpu",
        task_type="vision",
    )

    # Set explicit steps for analysis
    if hasattr(model, "max_steps"):
        model.max_steps = 50
    if hasattr(model, "eq_steps"):
        model.eq_steps = 50

    print(f"Model created: {spec.name}")

    # Load a sample input
    print("Loading a sample from Digits dataset...")
    dataset = get_vision_dataset("digits", train=True, flatten=True)
    x, y = dataset[0]  # Single sample (Tensor)
    x = x.unsqueeze(0)  # Add batch dimension -> [1, 64]

    # Initialize Analyzer
    analyzer = DynamicsAnalyzer(model, device="cpu")

    # 1. Analyze Convergence
    print("\nRunning Convergence Analysis...")
    data = analyzer.get_convergence_data(x, steps=50)

    fixed_point = data["fixed_point"]
    deltas = data["deltas"]

    print(f"Initial State Activity: {np.mean(np.abs(data['trajectory'][0])):.4f}")
    print(f"Final State Activity:   {np.mean(np.abs(fixed_point)):.4f}")
    print(f"Final Convergence Delta: {deltas[-1]:.6f}")

    # Plotting
    try:
        print("Generating Convergence Plot...")
        fig = analyzer.plot_convergence(
            x, steps=50, title="EqProp Dynamics (Untrained)"
        )
        fig.savefig("convergence_dynamics.png")
        print("Plot saved to 'convergence_dynamics.png'")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # 2. Gradient Alignment (Experimental)
    # Check if the model's updates align with backprop gradients
    # Note: On an initialized (untrained) model, alignment might be low or random,
    # but for EqProp it should be positive if beta is small.

    print("\nComputing Gradient Alignment...")
    # We need to set a criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Mock label
    y_target = torch.tensor([y]).long()

    alignment = analyzer.compute_gradient_alignment(x, y_target, criterion)
    print(f"Gradient Alignment (Cosine Similarity): {alignment:.4f}")

    if np.isnan(alignment):
        print(
            "(Alignment is NaN, likely because model does not support gradient_method switching or is not an EqProp model)"
        )


if __name__ == "__main__":
    main()
