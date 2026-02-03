"""
Automated Result Visualization

Generates publication-quality plots for experiment results using matplotlib and seaborn.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ResultVisualizer:
    """
    Generates standard plots for Bio-Plausible experiments.
    """

    def __init__(self, output_dir: Union[str, Path] = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": 300,
                "savefig.dpi": 300,
            }
        )

    def plot_lipschitz_trajectory(
        self,
        history: List[float],
        save_name: str = "lipschitz_trajectory.png",
        title: str = "Lipschitz Constant Dynamics",
    ):
        """
        Plot L(t) over training steps.
        Highlights the L=1 critical threshold.
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        steps = np.arange(len(history))
        ax.plot(steps, history, linewidth=2, label="Measured L")

        # Critical threshold
        ax.axhline(
            1.0,
            color="#e74c3c",
            linestyle="--",
            linewidth=1.5,
            label="L=1 (Contraction Limit)",
        )

        # Shade stable vs unstable regions
        ylim = ax.get_ylim()
        upper = max(max(history) * 1.1, 1.5)
        ax.fill_between(
            steps, 0, 1.0, color="#2ecc71", alpha=0.1, label="Stable Region"
        )
        ax.fill_between(
            steps, 1.0, upper, color="#e74c3c", alpha=0.05, label="Chaotic Region"
        )

        ax.set_ylim(0, upper)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Lipschitz Constant (L)")
        ax.set_title(title)
        ax.legend(loc="upper right")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_training_curves(
        self, metrics: Dict[str, List[float]], save_name: str = "training_curves.png"
    ):
        """
        Plot Loss and Accuracy curves side-by-side.
        Expects keys 'loss', 'accuracy', 'val_loss', 'val_accuracy' (optional).
        """
        has_acc = "accuracy" in metrics

        fig, axes = plt.subplots(
            1, 2 if has_acc else 1, figsize=(10 if has_acc else 5, 4)
        )
        if not has_acc:
            axes = [axes]

        # Plot Loss
        ax = axes[0]
        ax.plot(metrics["loss"], label="Train Loss")
        if "val_loss" in metrics:
            ax.plot(metrics["val_loss"], label="Val Loss", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()

        # Plot Accuracy
        if has_acc:
            ax = axes[1]
            ax.plot(metrics["accuracy"], label="Train Acc", color="orange")
            if "val_accuracy" in metrics:
                ax.plot(
                    metrics["val_accuracy"],
                    label="Val Acc",
                    linestyle="--",
                    color="darkorange",
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy")
            ax.legend()

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_memory_scaling(
        self,
        depths: List[int],
        backprop_mem: List[float],
        eqprop_mem: List[float],
        save_name: str = "memory_scaling.png",
    ):
        """
        Plot Memory Usage vs Depth (O(N) vs O(1)).
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(depths, backprop_mem, "o-", label="Backprop (BPTT)", color="#e74c3c")
        ax.plot(depths, eqprop_mem, "s-", label="EqProp (Implicit)", color="#2ecc71")

        ax.set_xlabel("Network Depth")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("Memory Wall: Backprop vs EqProp")
        ax.legend()
        ax.set_yscale("log")  # Usually log scale shows the order magnitude diff better

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_feature_alignment(
        self, angles: List[float], save_name: str = "alignment.png"
    ):
        """Plot alignment angle convergence."""
        fig, ax = plt.subplots(figsize=(6, 4))

        epochs = np.arange(len(angles))
        ax.plot(epochs, angles, linewidth=2, color="#9b59b6")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title("Feedback Alignment Convergence")
        ax.axhline(90, color="gray", linestyle=":", label="Orthogonal (90°)")
        ax.axhline(0, color="gray", linestyle="-", label="Aligned (0°)")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_leaderboard(
        self, data: List[Dict], task: str, save_name: Optional[str] = None
    ):
        """Bar chart of Top Accuracy per Model per Task."""
        if save_name is None:
            save_name = f"leaderboard_{task}.png"

        task_data = [d for d in data if d.get("task") == task]
        if not task_data:
            return

        models = sorted(list(set(d.get("model", "Unknown") for d in task_data)))
        best_accs = {}
        for m in models:
            m_data = [
                d.get("accuracy", 0) for d in task_data if d.get("model") == m
            ]
            if m_data:
                best_accs[m] = max(m_data)

        if not best_accs:
            return

        sorted_items = sorted(best_accs.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_items]
        vals = [x[1] for x in sorted_items]

        plt.figure(figsize=(10, 6), dpi=100)
        bars = plt.barh(names, vals, color="#4ecdc4")
        plt.title(f"Leaderboard: {task.upper()}", fontsize=14)
        plt.xlabel("Accuracy", fontsize=12)
        plt.xlim(0, 1.0)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2%}",
                va="center",
            )

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_tier_progress(
        self, data: List[Dict], save_name: str = "tier_progress.png"
    ):
        """Count of trials per tier."""
        from collections import defaultdict

        tiers = ["smoke", "shallow", "standard", "deep"]
        counts = defaultdict(int)
        for d in data:
            t = d.get("tier", "unknown")
            counts[t] += 1

        plt.figure(figsize=(8, 5), dpi=100)
        x = range(len(tiers))
        y = [counts[t] for t in tiers]
        plt.bar(x, y, color="#ff6b6b")
        plt.xticks(x, [t.title() for t in tiers])
        plt.title("Experimental Progress (Trial Counts)", fontsize=14)
        plt.ylabel("Number of Trials", fontsize=12)
        plt.grid(axis="y", alpha=0.3)
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_hyperparam_correlations(
        self, data: List[Dict], save_name_prefix: str = "impact_"
    ):
        """Scatter plots of Hyperparams vs Accuracy."""
        params = ["learning_rate", "beta", "weight_decay"]
        saved_files = []

        for param in params:
            vals = []
            accs = []
            for d in data:
                if param in d and isinstance(d[param], (int, float)):
                    vals.append(d[param])
                    accs.append(d.get("accuracy", 0))

            if not vals:
                continue

            plt.figure(figsize=(8, 5), dpi=100)
            plt.scatter(vals, accs, alpha=0.6, c=accs, cmap="viridis")
            plt.title(f"Impact of {param}", fontsize=14)
            plt.xlabel(param, fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            if param == "learning_rate":
                plt.xscale("log")
            plt.colorbar(label="Accuracy")
            plt.grid(True, alpha=0.3)
            save_path = self.output_dir / f"{save_name_prefix}{param}.png"
            plt.savefig(save_path)
            plt.close()
            saved_files.append(str(save_path))
        return saved_files

    def plot_pareto_frontier(
        self, data: List[Dict], save_name: str = "pareto_frontier.png"
    ):
        """Pareto Frontier: Accuracy vs Parameters."""
        plt.figure(figsize=(10, 6), dpi=100)

        models = list(set(d.get("model", "Unknown") for d in data))

        for model in models:
            m_data = [d for d in data if d.get("model") == model]
            x = [d.get("params", 0) for d in m_data]
            y = [d.get("accuracy", 0) for d in m_data]

            # Simple scatter
            plt.scatter(x, y, label=model, alpha=0.7)

        plt.title("Efficiency Frontier (Accuracy vs Scale)", fontsize=14)
        plt.xlabel("Parameters (Millions)", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_significance_matrix(
        self,
        p_values: np.ndarray,
        labels: List[str],
        save_name: str = "significance_matrix.png",
    ):
        """Heatmap of P-values between models."""
        plt.figure(figsize=(10, 8), dpi=100)
        sns.heatmap(
            p_values,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            cmap="Blues_r",
            vmin=0,
            vmax=0.05,
        )
        plt.title("Statistical Significance (P-Values)", fontsize=14)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)
