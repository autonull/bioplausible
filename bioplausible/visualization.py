"""
Automated Result Visualization

Generates publication-quality plots for experiment results using matplotlib and seaborn.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")


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

    def plot_convergence_curves(
        self, trajectories: List[Any], save_name: str = "convergence_curves.png"
    ):
        """
        Plot Accuracy vs Epoch for top trajectories.
        Expects trajectories to have 'model_name', 'task_name',
        and 'checkpoints' (list of objects with epoch, val_acc/test_acc).
        """
        # Group by task
        from collections import defaultdict

        task_trajectories = defaultdict(list)
        for t in trajectories:
            task_trajectories[t.task_name].append(t)

        saved_files = []
        for task, trajs in task_trajectories.items():
            # Pick best trajectory per model
            best_per_model = {}
            for t in trajs:
                # Calculate max accuracy in this trajectory
                if not t.checkpoints:
                    continue
                max_acc = max(ckpt.val_acc for ckpt in t.checkpoints)
                if (
                    t.model_name not in best_per_model
                    or max_acc > best_per_model[t.model_name][1]
                ):
                    best_per_model[t.model_name] = (t, max_acc)

            if not best_per_model:
                continue

            # Plot
            plt.figure(figsize=(10, 6), dpi=100)
            for model, (traj, _) in best_per_model.items():
                epochs = [ckpt.epoch for ckpt in traj.checkpoints]
                accs = [ckpt.val_acc for ckpt in traj.checkpoints]
                plt.plot(epochs, accs, label=model, alpha=0.8, linewidth=2)

            plt.title(f"Convergence Trajectories: {task.upper()}", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Validation Accuracy", fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            task_save_name = f"convergence_curves_{task}.png"
            save_path = self.output_dir / task_save_name
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            saved_files.append(str(save_path))

        return saved_files

    def plot_learning_dynamics(
        self, trajectories: List[Any], save_name: str = "learning_dynamics.png"
    ):
        """
        Plot Learning Rate vs Epoch for top trajectories.
        """
        # Group by task
        from collections import defaultdict

        task_trajectories = defaultdict(list)
        for t in trajectories:
            task_trajectories[t.task_name].append(t)

        saved_files = []
        for task, trajs in task_trajectories.items():
            # Pick best trajectory per model
            best_per_model = {}
            for t in trajs:
                if not t.checkpoints:
                    continue
                max_acc = max(ckpt.val_acc for ckpt in t.checkpoints)
                if (
                    t.model_name not in best_per_model
                    or max_acc > best_per_model[t.model_name][1]
                ):
                    best_per_model[t.model_name] = (t, max_acc)

            if not best_per_model:
                continue

            # Plot
            plt.figure(figsize=(10, 6), dpi=100)
            for model, (traj, _) in best_per_model.items():
                epochs = []
                lrs = []
                for ckpt in traj.checkpoints:
                    if hasattr(ckpt, "learning_rate"):
                        epochs.append(ckpt.epoch)
                        lrs.append(ckpt.learning_rate)

                if lrs:
                    plt.plot(epochs, lrs, label=model, alpha=0.8, linewidth=2)

            plt.title(f"Learning Rate Schedule: {task.upper()}", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Learning Rate", fontsize=12)
            plt.yscale(
                "log"
            )  # LR usually varies logarithmically or we want to see decay
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
            plt.grid(True, alpha=0.3, which="both", ls="-")
            plt.tight_layout()

            task_save_name = f"learning_dynamics_{task}.png"
            save_path = self.output_dir / task_save_name
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            saved_files.append(str(save_path))

        return saved_files

    def plot_sample_complexity(
        self, trajectories: List[Any], save_name: str = "sample_complexity.png"
    ):
        """
        Plot Accuracy vs Samples Seen.
        """
        # Group by task
        from collections import defaultdict

        task_trajectories = defaultdict(list)
        for t in trajectories:
            task_trajectories[t.task_name].append(t)

        saved_files = []
        for task, trajs in task_trajectories.items():
            # Pick best trajectory per model
            best_per_model = {}
            for t in trajs:
                # Calculate max accuracy in this trajectory
                if not t.checkpoints:
                    continue
                max_acc = max(ckpt.val_acc for ckpt in t.checkpoints)
                if (
                    t.model_name not in best_per_model
                    or max_acc > best_per_model[t.model_name][1]
                ):
                    best_per_model[t.model_name] = (t, max_acc)

            if not best_per_model:
                continue

            # Plot
            plt.figure(figsize=(10, 6), dpi=100)
            for model, (traj, _) in best_per_model.items():
                samples = []
                accs = []
                for ckpt in traj.checkpoints:
                    if hasattr(ckpt, "samples_seen") and ckpt.samples_seen > 0:
                        samples.append(ckpt.samples_seen)
                        accs.append(ckpt.val_acc)
                    else:
                        # Fallback if samples_seen is 0 (legacy data)
                        # We skip plotting samples for this legacy trail
                        pass

                if len(samples) == len(accs) and samples:
                    plt.plot(samples, accs, label=model, alpha=0.8, linewidth=2)

            plt.title(f"Sample Complexity: {task.upper()}", fontsize=14)
            plt.xlabel("Training Samples Seen", fontsize=12)
            plt.ylabel("Validation Accuracy", fontsize=12)
            plt.xscale("log")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
            plt.grid(True, alpha=0.3, which="both", ls="-")
            plt.tight_layout()

            task_save_name = f"sample_complexity_{task}.png"
            save_path = self.output_dir / task_save_name
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            saved_files.append(str(save_path))

        return saved_files

    def plot_family_leaderboard(
        self, data: List[Dict], save_name: str = "leaderboard_families.png"
    ):
        """Bar chart of Mean Accuracy per Algorithm Family."""
        from collections import defaultdict

        family_accs = defaultdict(list)
        for d in data:
            f = d.get("family", "Other")
            family_accs[f].append(d.get("accuracy", 0))

        if not family_accs:
            return ""

        families = []
        means = []
        stds = []

        for f, accs in family_accs.items():
            families.append(f)
            means.append(np.mean(accs))
            stds.append(np.std(accs) if len(accs) > 1 else 0)

        # Sort
        sorted_indices = np.argsort(means)
        families = [families[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6), dpi=100)
        bars = plt.barh(families, means, xerr=stds, color="#6c5ce7", capsize=5)
        plt.title("Algorithm Family Performance (Mean Accuracy)", fontsize=14)
        plt.xlabel("Mean Accuracy", fontsize=12)
        plt.xlim(0, 1.05)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f"{width:.2%}"
            if stds[i] > 0:
                label += f" ±{stds[i]:.2%}"
            plt.text(
                width + 0.02,
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

            elif metric == "compound_efficiency":
                # Accuracy / (Params * Epochs)
                params = max(d.get("params", 0), 0.001)
                epochs = max(d.get("epochs", 1), 1)
                val = d.get("accuracy", 0) / (params * epochs)

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
            # Prefer CI if available, else std
            xerr = [d.get("accuracy_ci_95", d.get("accuracy_std", 0)) for d in entries]
            if all(x == 0 for x in xerr):
                xerr = None

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

        if metric == "efficiency":
            title_metric = "Efficiency (Acc / M-Params)"
        elif metric == "compound_efficiency":
            title_metric = "Compound Efficiency (Acc / (M-Params * Epochs))"
        else:
            title_metric = "Score / Accuracy (Mean)"

        plt.title(
            f"Leaderboard ({metric.replace('_', ' ').title()}): {task.upper()}",
            fontsize=14,
        )
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
                (
                    width + (0.02 if not xerr else xerr[i] + 0.02)
                    if metric == "accuracy"
                    else width + (max_val * 0.02)
                ),
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

        # Identify unique task/tier combinations
        combinations = set()
        for d in data:
            t = d.get("task", "unknown")
            ti = d.get("tier", "unknown")
            combinations.add((t, ti))

        for task, tier in combinations:
            # Filter data for this combo
            subset = [
                d for d in data if d.get("task") == task and d.get("tier") == tier
            ]
            if len(subset) < 10:
                continue

            # Create folder structure
            combo_dir = self.output_dir / task / tier
            combo_dir.mkdir(parents=True, exist_ok=True)

            for param in params:
                vals = []
                accs = []
                for d in subset:
                    if param in d and isinstance(d[param], (int, float)):
                        vals.append(d[param])
                        accs.append(d.get("accuracy", 0))

                if not vals:
                    continue

                plt.figure(figsize=(8, 5), dpi=100)
                plt.scatter(vals, accs, alpha=0.6, c=accs, cmap="viridis")

                title = f"Impact of {param}: {task} ({tier})"
                plt.title(title, fontsize=14)
                plt.xlabel(param, fontsize=12)
                plt.ylabel("Accuracy", fontsize=12)

                if param == "learning_rate":
                    plt.xscale("log")

                plt.colorbar(label="Accuracy")
                plt.grid(True, alpha=0.3)

                save_path = combo_dir / f"{save_name_prefix}{param}.png"
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
            plt.scatter(x, y, label=model, alpha=0.8, edgecolors="w", s=60)

        plt.title(
            f"Efficiency Frontier (Top 50% Performers, Acc >= {median_acc:.2%})",
            fontsize=14,
        )
        plt.xlabel("Parameters (Millions)", fontsize=12)
        plt.ylabel("Score / Accuracy", fontsize=12)

        # Move legend outside to avoid clutter
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  # Adjust layout to make room for legend

        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return str(save_path)

    def plot_convergence_speed(
        self, data: List[Dict], save_name: str = "convergence_speed.png"
    ):
        """Plot Convergence Speed (1/Epochs) vs Final Accuracy."""
        plt.figure(figsize=(10, 6), dpi=100)

        # Filter data with iteration_time or time
        valid_data = [
            d
            for d in data
            if d.get("iteration_time", 0) > 0 and d.get("accuracy", 0) > 0.1
        ]
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return str(save_path)

    def plot_sensitivity_heatmap(
        self,
        sensitivity: Dict[str, Dict[str, float]],
        save_name: str = "sensitivity_heatmap.png",
    ):
        """Heatmap of Hyperparameter Sensitivity per Model."""
        # Convert to matrix
        models = sorted(sensitivity.keys())
        all_params = set()
        for m in models:
            all_params.update(sensitivity[m].keys())
        params = sorted(list(all_params))

        if not models or not params:
            return ""

        matrix = np.zeros((len(models), len(params)))
        for i, m in enumerate(models):
            for j, p in enumerate(params):
                matrix[i, j] = sensitivity[m].get(p, 0.0)

        plt.figure(figsize=(12, 8), dpi=100)
        sns.heatmap(
            matrix,
            xticklabels=params,
            yticklabels=models,
            annot=True,
            fmt=".2f",
            cmap="Reds",
        )
        plt.title("Hyperparameter Sensitivity (Variance Index)", fontsize=14)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_hyperparam_heatmap(
        self,
        data: List[Dict],
        param_x: str,
        param_y: str,
        metric: str = "accuracy",
        save_name: Optional[str] = None,
    ):
        """2D Heatmap of metric vs two hyperparameters."""
        # Extract x, y, z
        xs, ys, zs = [], [], []
        for d in data:
            if param_x in d and param_y in d:
                xs.append(d[param_x])
                ys.append(d[param_y])
                zs.append(d.get(metric, 0))

        if not xs:
            return ""

        if save_name is None:
            save_name = f"heatmap_{param_x}_{param_y}.png"

        plt.figure(figsize=(8, 6), dpi=100)
        # Use hexbin for density/averaging
        # gridsize depends on data points
        gridsize = max(10, int(np.sqrt(len(xs))))
        hb = plt.hexbin(
            xs, ys, C=zs, gridsize=gridsize, cmap="viridis", reduce_C_function=np.mean
        )
        plt.colorbar(hb, label=f"Mean {metric}")
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.title(f"{metric} Heatmap: {param_x} vs {param_y}")
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_task_difficulty(
        self, data: List[Dict], save_name: str = "task_difficulty.png"
    ):
        """Plot Mean Accuracy vs Variance for each task."""
        from collections import defaultdict

        task_stats = defaultdict(list)
        for d in data:
            task_stats[d["task"]].append(d.get("accuracy", 0))

        if not task_stats:
            return ""

        means = []
        stds = []
        labels = []

        for t, accs in task_stats.items():
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            labels.append(t)

        plt.figure(figsize=(10, 6), dpi=100)
        plt.scatter(means, stds, s=100, alpha=0.7)

        for i, txt in enumerate(labels):
            plt.annotate(
                txt, (means[i], stds[i]), xytext=(5, 5), textcoords="offset points"
            )

        plt.title("Task Difficulty Analysis", fontsize=14)
        plt.xlabel("Mean Accuracy (Ease)", fontsize=12)
        plt.ylabel("Standard Deviation (Instability)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_hexbin(
        self, data: List[Dict], x_col: str, y_col: str, save_name: Optional[str] = None
    ):
        """Generates a hexbin plot for dense data."""
        xs = [
            d.get(x_col)
            for d in data
            if d.get(x_col) is not None and d.get(y_col) is not None
        ]
        ys = [
            d.get(y_col)
            for d in data
            if d.get(x_col) is not None and d.get(y_col) is not None
        ]

        if not xs:
            return ""

        if save_name is None:
            save_name = f"hexbin_{x_col}_{y_col}.png"

        plt.figure(figsize=(8, 6), dpi=100)
        gridsize = max(10, int(np.sqrt(len(xs))))
        hb = plt.hexbin(xs, ys, gridsize=gridsize, cmap="Blues", mincnt=1)
        plt.colorbar(hb, label="Count")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Density Plot: {x_col} vs {y_col}")
        plt.tight_layout()

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
