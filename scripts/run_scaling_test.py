import os

import numpy as np
import pandas as pd

from bioplausible.analysis.scaling import (
    compute_compute_optimal,
    fit_power_law,
    plot_scaling_curves,
)


def main():
    print("Testing Phase 1.4: Scaling Law Analysis...")

    # Mock data representing parameter sweeps for two fictional models
    mock_data = {"model": [], "param_count": [], "val_loss": [], "val_accuracy": []}

    param_counts = [10000, 30000, 100000, 300000, 1000000]

    # Model A: Good scaling
    for N in param_counts:
        # a * N^-b + noise
        loss = 100.0 * (N**-0.3) * np.random.uniform(0.9, 1.1)
        acc = 1.0 - loss
        mock_data["model"].append("Model_A")
        mock_data["param_count"].append(N)
        mock_data["val_loss"].append(loss)
        mock_data["val_accuracy"].append(acc)

    # Model B: Poor scaling
    for N in param_counts:
        loss = 50.0 * (N**-0.15) * np.random.uniform(0.9, 1.1)
        acc = 1.0 - loss
        mock_data["model"].append("Model_B")
        mock_data["param_count"].append(N)
        mock_data["val_loss"].append(loss)
        mock_data["val_accuracy"].append(acc)

    df = pd.DataFrame(mock_data)

    # 1. Test compute optimal
    optimal = compute_compute_optimal(df)
    print("Compute Optimal Frontier:")
    print(optimal)

    # 2. Test power-law curve fitting
    for model in ["Model_A", "Model_B"]:
        m_df = optimal[optimal["model"] == model]
        if len(m_df) == 0:
            continue
        a, b = fit_power_law(m_df["param_count"].tolist(), m_df["val_loss"].tolist())
        print(f"{model} fit: L = {a:.4f} * N^(-{b:.4f})")
        assert not np.isnan(a) and not np.isnan(b)

    # 3. Test plotting
    output_dir = "results/scaling_test"
    os.makedirs(output_dir, exist_ok=True)
    try:
        fig = plot_scaling_curves(optimal, metric="val_loss")
        fig.savefig(f"{output_dir}/scaling_curves_loss.png")
        print("\nSaved scaling_curves_loss.png")

        fig2 = plot_scaling_curves(optimal, metric="val_accuracy")
        fig2.savefig(f"{output_dir}/scaling_curves_acc.png")
        print("Saved scaling_curves_acc.png")

    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
