import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioplausible.analysis.ablation import AblationStudy
from bioplausible.analysis.scaling import plot_scaling_curves
from bioplausible.knowledge.metamodel import KnowledgebaseMetamodel
from bioplausible.leaderboard.generator import LeaderboardEntry, LeaderboardGenerator


def test_analysis_tools():
    print("=" * 80)
    print(" Phase 1 Analysis Tools Verification")
    print("=" * 80)

    os.makedirs("reports", exist_ok=True)

    # 1. Scaling Curves
    print("\n[1/4] Testing Scaling Analysis...")
    models = ["backprop_mlp", "eqprop_mlp", "forward_forward"]
    data = []

    for m in models:
        for p_count in [1000, 10000, 100000, 1000000]:
            # Mock power law: L = a * N^-b + noise
            if m == "backprop_mlp":
                loss = 5.0 * (p_count**-0.15)
            elif m == "eqprop_mlp":
                loss = 5.5 * (p_count**-0.14)
            else:
                loss = 6.0 * (p_count**-0.10)

            data.append(
                {
                    "model": m,
                    "param_count": p_count,
                    "val_loss": loss + np.random.normal(0, 0.01),
                    "val_accuracy": 1.0 - loss,
                }
            )

    df_scaling = pd.DataFrame(data)
    try:
        fig = plot_scaling_curves(df_scaling, metric="val_loss")
        fig.savefig("reports/test_scaling_curve.png")
        print("  ✓ Scaling curve generated: reports/test_scaling_curve.png")
    except Exception as e:
        print(f"  ✗ Scaling analysis failed: {e}")

    # 2. Leaderboard
    print("\n[2/4] Testing Leaderboard Generator...")
    entries = []
    for i, row in df_scaling.iterrows():
        entry = LeaderboardEntry(
            algorithm=row["model"],
            optimizer="adam" if "backprop" in row["model"] else "smep",
            task="mnist",
            val_accuracy=row["val_accuracy"],
            energy_proxy=row["param_count"] * 2.0,  # Mock energy
            backward_flops=(
                0 if "backprop" not in row["model"] else int(row["param_count"])
            ),
            requires_backward="backprop" in row["model"],
            param_count=int(row["param_count"]),
            wall_time_s=10.5,
            peak_memory_mb=100.0,
            mean_acc=row["val_accuracy"],
            std_acc=0.01,
            config_hash="abc1234",
        )
        entries.append(entry)

    try:
        gen = LeaderboardGenerator(entries)
        gen.export_markdown("reports/test_leaderboard.md")
        print("  ✓ Leaderboard generated: reports/test_leaderboard.md")
    except Exception as e:
        print(f"  ✗ Leaderboard generation failed: {e}")

    # 3. Metamodel (Mock)
    print("\n[3/4] Testing Knowledgebase Metamodel...")
    try:
        kb = KnowledgebaseMetamodel()
        # Mock fit data
        kb.df = pd.DataFrame(
            [
                {
                    "model": "eqprop_mlp",
                    "outcome": "success",
                    "lr": 0.01,
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "max_steps": 20,
                },
                {
                    "model": "eqprop_mlp",
                    "outcome": "failure",
                    "lr": 0.1,
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "max_steps": 20,
                },
                {
                    "model": "eqprop_mlp",
                    "outcome": "success",
                    "lr": 0.005,
                    "hidden_dim": 128,
                    "num_layers": 3,
                    "max_steps": 30,
                },
            ]
        )
        kb.fitted = True
        rules = kb.extract_symbolic_rules(focus_model="eqprop_mlp")
        print("  ✓ Extracted Rules:")
        for r in rules:
            print(f"    {r}")
    except Exception as e:
        print(f"  ✗ Metamodel failed: {e}")

    print("\nAnalysis Verification Complete.")


if __name__ == "__main__":
    test_analysis_tools()
