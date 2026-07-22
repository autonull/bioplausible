"""
Bioplausible Visualization Utilities

Visualization tools for experiments, models, and results.

Features:
- Training curve plotting
- Comparison charts
- Confusion matrices
- Model architecture visualization
- Results dashboard generation
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class TrainingVisualizer:
    """
    Visualize training progress and results.

    Example usage:
        viz = TrainingVisualizer()

        # Plot training curves
        viz.plot_training_curve(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path='training_curve.png',
        )

        # Plot comparison
        viz.plot_comparison(
            results=[result1, result2, result3],
            metric='val_accuracy',
            save_path='comparison.png',
        )
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        self.style = style
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except OSError, IOError:
                pass  # Use default style

    def plot_training_curve(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_accuracies: Optional[List[float]] = None,
        val_accuracies: Optional[List[float]] = None,
        title: str = "Training Progress",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot training curves.

        Args:
            train_losses: Training losses per epoch.
            val_losses: Validation losses per epoch.
            train_accuracies: Training accuracies per epoch.
            val_accuracies: Validation accuracies per epoch.
            title: Plot title.
            save_path: Path to save figure.
            show: Show plot.

        Returns:
            Matplotlib figure.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required: pip install matplotlib")

        epochs = list(range(1, len(train_losses) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1 = axes[0]
        ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
        if val_losses:
            ax1.plot(epochs, val_losses, "r--", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{title} - Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2 = axes[1]
        if train_accuracies:
            ax2.plot(epochs, train_accuracies, "b-", label="Train Acc", linewidth=2)
        if val_accuracies:
            ax2.plot(epochs, val_accuracies, "r--", label="Val Acc", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"{title} - Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_comparison(
        self,
        results: List[Any],
        metric: str = "val_accuracy",
        title: str = "Optimizer Comparison",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot comparison of multiple results.

        Args:
            results: List of ExperimentResult objects.
            metric: Metric to compare ('val_accuracy', 'train_loss', etc.).
            title: Plot title.
            save_path: Path to save figure.
            show: Show plot.

        Returns:
            Matplotlib figure.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required: pip install matplotlib")

        names = [r.optimizer_name for r in results]
        values = [getattr(r, metric) for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=1.5)

        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Optimizer")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x labels if needed
        if len(names) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_speed_accuracy_tradeoff(
        self,
        results: List[Any],
        title: str = "Speed vs Accuracy Trade-off",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot speed vs accuracy trade-off.

        Args:
            results: List of ExperimentResult objects.
            title: Plot title.
            save_path: Path to save figure.
            show: Show plot.

        Returns:
            Matplotlib figure.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required: pip install matplotlib")

        speeds = [r.steps_per_second for r in results]
        accuracies = [r.val_accuracy for r in results]
        names = [r.optimizer_name for r in results]

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(
            speeds,
            accuracies,
            s=200,
            alpha=0.7,
            c=range(len(names)),
            cmap="viridis",
            edgecolors="black",
            linewidth=1.5,
        )

        # Add labels
        for i, name in enumerate(names):
            ax.annotate(
                name,
                (speeds[i], accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_xlabel("Training Speed (steps/second)")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            class_names: Names of classes.
            title: Plot title.
            save_path: Path to save figure.
            show: Show plot.

        Returns:
            Matplotlib figure.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required: pip install matplotlib")

        # Compute confusion matrix
        classes = sorted(set(y_true) | set(y_pred))
        n_classes = len(classes)

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[classes.index(t), classes.index(p)] += 1

        if class_names is None:
            class_names = [str(c) for c in classes]

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig


class ResultsDashboard:
    """
    Generate HTML dashboard for experiment results.

    Example usage:
        dashboard = ResultsDashboard()

        dashboard.add_results(results)
        dashboard.generate('results_dashboard.html')
    """

    def __init__(self):
        self.results = []
        self.html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Bioplausible Experiment Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}  # noqa: E501
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}  # noqa: E501
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .metric {{ font-weight: bold; color: #4CAF50; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}  # noqa: E501
        .card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }}  # noqa: E501
        .card-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .card-label {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Bioplausible Experiment Results</h1>
        {content}
    </div>
</body>
</html>
"""

    def add_results(self, results: List[Any]) -> None:
        """Add experiment results to dashboard."""
        self.results.extend(results)

    def generate(self, output_path: str) -> str:
        """
        Generate HTML dashboard.

        Args:
            output_path: Path to save HTML file.

        Returns:
            Path to generated file.
        """
        content = self._generate_content()
        html = self.html_template.format(content=content)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _generate_content(self) -> str:
        """Generate HTML content from results."""
        if not self.results:
            return "<p>No results to display.</p>"

        # Summary cards
        best_result = max(self.results, key=lambda r: r.val_accuracy)
        fastest_result = max(self.results, key=lambda r: r.steps_per_second)

        summary = f"""
        <h2>Summary</h2>
        <div class="summary">
            <div class="card">
                <div class="card-label">Best Accuracy</div>
                <div class="card-value">{best_result.val_accuracy:.2f}%</div>
                <div>{best_result.optimizer_name}</div>
            </div>
            <div class="card">
                <div class="card-label">Fastest Training</div>
                <div class="card-value">{fastest_result.steps_per_second:.1f}</div>
                <div>steps/second</div>
            </div>
            <div class="card">
                <div class="card-label">Experiments</div>
                <div class="card-value">{len(self.results)}</div>
                <div>total runs</div>
            </div>
        </div>
        """

        # Results table
        table = """
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Optimizer</th>
                    <th>Val Accuracy</th>
                    <th>Train Accuracy</th>
                    <th>Speed (steps/s)</th>
                    <th>Parameters</th>
                </tr>
            </thead>
            <tbody>
        """

        for r in sorted(self.results, key=lambda x: x.val_accuracy, reverse=True):
            table += f"""
                <tr>
                    <td>{r.model_name}</td>
                    <td>{r.optimizer_name}</td>
                    <td class="metric">{r.val_accuracy:.2f}%</td>
                    <td>{r.train_accuracy:.2f}%</td>
                    <td>{r.steps_per_second:.1f}</td>
                    <td>{r.num_parameters:,}</td>
                </tr>
            """

        table += "</tbody></table>"

        return summary + table


def visualize_results(
    results: List[Any],
    output_dir: str = "./visualizations",
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate all visualizations for experiment results.

    Args:
        results: List of ExperimentResult objects.
        output_dir: Output directory for visualizations.
        show: Show plots.

    Returns:
        Dict of visualization paths.
    """
    if not HAS_MATPLOTLIB:
        return {"error": "matplotlib not installed"}

    os.makedirs(output_dir, exist_ok=True)
    viz = TrainingVisualizer()
    paths = {}

    # Comparison chart
    paths["comparison"] = viz.plot_comparison(
        results,
        metric="val_accuracy",
        title="Optimizer Comparison",
        save_path=os.path.join(output_dir, "comparison.png"),
        show=show,
    )

    # Speed vs accuracy
    paths["tradeoff"] = viz.plot_speed_accuracy_tradeoff(
        results,
        title="Speed vs Accuracy Trade-off",
        save_path=os.path.join(output_dir, "tradeoff.png"),
        show=show,
    )

    # Dashboard
    dashboard = ResultsDashboard()
    dashboard.add_results(results)
    paths["dashboard"] = dashboard.generate(os.path.join(output_dir, "dashboard.html"))

    return paths


__all__ = [
    "TrainingVisualizer",
    "ResultsDashboard",
    "visualize_results",
]
