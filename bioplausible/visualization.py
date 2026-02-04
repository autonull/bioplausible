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
        self,
        data: List[Dict],
        task: str,
        save_name: Optional[str] = None,
        use_std: bool = False,
        metric: str = "accuracy",
    ):
        """Bar chart of Metric per Model per Task."""
        if save_name is None:
            save_name = f"leaderboard_{task}_{metric}.png"

        task_data = [d for d in data if d.get("task") == task]
        if not task_data:
            return

        # Find best config for each model based on metric
        best_entries = {}
        for d in task_data:
            model = d.get("model", "Unknown")
            val = d.get(metric, 0)
            if metric == "efficiency":
                # Calculate efficiency: Accuracy / Params (Millions)
                # Avoid division by zero
                params = max(d.get("params", 0), 0.001)
                val = d.get("accuracy", 0) / params

            if model not in best_entries:
                best_entries[model] = {"entry": d, "val": val}
            else:
                if val > best_entries[model]["val"]:
                    best_entries[model] = {"entry": d, "val": val}

        if not best_entries:
            return

        sorted_items = sorted(best_entries.items(), key=lambda x: x[1]["val"])
        names = [k for k, v in sorted_items]
        vals = [v["val"] for k, v in sorted_items]
        entries = [v["entry"] for k, v in sorted_items]

        # Prepare error bars (only for accuracy currently)
        xerr = None
        if use_std and metric == "accuracy":
            stds = [d.get("accuracy_std", 0) for d in entries]
            if any(s > 0 for s in stds):
                xerr = stds

        plt.figure(figsize=(10, 6), dpi=100)
        color = "#4ecdc4" if metric == "accuracy" else "#ff9f43"
        bars = plt.barh(
            names,
            vals,
            xerr=xerr,
            color=color,
            capsize=5,
            error_kw={"ecolor": "gray", "alpha": 0.7},
        )

        title_metric = "Efficiency (Acc / M-Params)" if metric == "efficiency" else "Score / Accuracy (Mean)"
        plt.title(f"Leaderboard ({metric.title()}): {task.upper()}", fontsize=14)
        plt.xlabel(title_metric, fontsize=12)

        # Adjust xlim
        max_val = max(vals)
        if metric == "accuracy":
            plt.xlim(0, 1.05)
        else:
            plt.xlim(0, max_val * 1.15)

        plt.grid(axis="x", linestyle="--", alpha=0.7)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            if metric == "accuracy":
                label = f"{width:.2%}"
            else:
                label = f"{width:.2f}"

            if xerr:
                std = xerr[i]
                if std > 0:
                    label += f" ±{std:.2%}"

            plt.text(
                width + (0.02 if not xerr else xerr[i] + 0.02) if metric == "accuracy" else width + (max_val*0.02),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=9,
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
        # Removed 'beta' as requested (not helpful)
        params = ["learning_rate", "weight_decay", "hidden_dim", "num_layers"]
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
        """Pareto Frontier: Accuracy vs Parameters. Shows only top 50% performers."""
        plt.figure(figsize=(10, 6), dpi=100)

        # Filter low performers globally to reduce pollution
        accs = [d.get("accuracy", 0) for d in data]
        if not accs:
            return str(self.output_dir / save_name)

        median_acc = np.median(accs)
        filtered_data = [d for d in data if d.get("accuracy", 0) >= median_acc]

        models = sorted(list(set(d.get("model", "Unknown") for d in filtered_data)))

        for model in models:
            m_data = [d for d in filtered_data if d.get("model") == model]
            if not m_data:
                continue

            x = [d.get("params", 0) for d in m_data]
            y = [d.get("accuracy", 0) for d in m_data]

            # Simple scatter with slightly larger points
            plt.scatter(x, y, label=model, alpha=0.8, edgecolors='w', s=60)

        plt.title(f"Efficiency Frontier (Top 50% Performers, Acc >= {median_acc:.2%})", fontsize=14)
        plt.xlabel("Parameters (Millions)", fontsize=12)
        plt.ylabel("Score / Accuracy", fontsize=12)

        # Move legend outside to avoid clutter
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        plt.tight_layout() # Adjust layout to make room for legend

        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return str(save_path)

    def plot_convergence_speed(
        self, data: List[Dict], save_name: str = "convergence_speed.png"
    ):
        """Plot Convergence Speed (1/Epochs) vs Final Accuracy."""
        plt.figure(figsize=(10, 6), dpi=100)

        # Filter data with iteration_time or time
        valid_data = [d for d in data if d.get("iteration_time", 0) > 0 and d.get("accuracy", 0) > 0.1]
        if not valid_data:
            return ""

        models = sorted(list(set(d.get("model", "Unknown") for d in valid_data)))

        for model in models:
            m_data = [d for d in valid_data if d.get("model") == model]
            # Speed proxy: Accuracy / Time (Acc per second) or just Raw Time?
            # User asked for "Fast Learners".
            # Let's plot Accuracy vs Iteration Time (log scale x)

            x = [d.get("iteration_time", 0) for d in m_data]
            y = [d.get("accuracy", 0) for d in m_data]

            plt.scatter(x, y, label=model, alpha=0.7)

        plt.title("Convergence Speed: Accuracy vs Iteration Cost", fontsize=14)
        plt.xlabel("Time per Iteration (s)", fontsize=12)
        plt.ylabel("Final Accuracy", fontsize=12)
        plt.xscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
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
