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
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
        })

    def plot_lipschitz_trajectory(
        self,
        history: List[float],
        save_name: str = "lipschitz_trajectory.png",
        title: str = "Lipschitz Constant Dynamics"
    ):
        """
        Plot L(t) over training steps.
        Highlights the L=1 critical threshold.
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        steps = np.arange(len(history))
        ax.plot(steps, history, linewidth=2, label='Measured L')

        # Critical threshold
        ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='L=1 (Contraction Limit)')

        # Shade stable vs unstable regions
        ylim = ax.get_ylim()
        upper = max(max(history) * 1.1, 1.5)
        ax.fill_between(steps, 0, 1.0, color='#2ecc71', alpha=0.1, label='Stable Region')
        ax.fill_between(steps, 1.0, upper, color='#e74c3c', alpha=0.05, label='Chaotic Region')

        ax.set_ylim(0, upper)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Lipschitz Constant (L)")
        ax.set_title(title)
        ax.legend(loc='upper right')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_training_curves(
        self,
        metrics: Dict[str, List[float]],
        save_name: str = "training_curves.png"
    ):
        """
        Plot Loss and Accuracy curves side-by-side.
        Expects keys 'loss', 'accuracy', 'val_loss', 'val_accuracy' (optional).
        """
        has_acc = 'accuracy' in metrics

        fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(10 if has_acc else 5, 4))
        if not has_acc: axes = [axes]

        # Plot Loss
        ax = axes[0]
        ax.plot(metrics['loss'], label='Train Loss')
        if 'val_loss' in metrics:
            ax.plot(metrics['val_loss'], label='Val Loss', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()

        # Plot Accuracy
        if has_acc:
            ax = axes[1]
            ax.plot(metrics['accuracy'], label='Train Acc', color='orange')
            if 'val_accuracy' in metrics:
                ax.plot(metrics['val_accuracy'], label='Val Acc', linestyle='--', color='darkorange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy')
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
        save_name: str = "memory_scaling.png"
    ):
        """
        Plot Memory Usage vs Depth (O(N) vs O(1)).
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(depths, backprop_mem, 'o-', label='Backprop (BPTT)', color='#e74c3c')
        ax.plot(depths, eqprop_mem, 's-', label='EqProp (Implicit)', color='#2ecc71')

        ax.set_xlabel("Network Depth")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("Memory Wall: Backprop vs EqProp")
        ax.legend()
        ax.set_yscale('log') # Usually log scale shows the order magnitude diff better

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_feature_alignment(
        self,
        angles: List[float],
        save_name: str = "alignment.png"
    ):
        """Plot alignment angle convergence."""
        fig, ax = plt.subplots(figsize=(6, 4))

        epochs = np.arange(len(angles))
        ax.plot(epochs, angles, linewidth=2, color='#9b59b6')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title("Feedback Alignment Convergence")
        ax.axhline(90, color='gray', linestyle=':', label='Orthogonal (90°)')
        ax.axhline(0, color='gray', linestyle='-', label='Aligned (0°)')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path)
        plt.close()
        return str(save_path)
