"""
MEP Benchmark Visualization

Generate publication-quality plots for benchmark results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class BenchmarkVisualizer:
    """Generate visualizations for benchmark results."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

    @staticmethod
    def _format_optimizer_name(name: str) -> str:
        """Format optimizer name for display."""
        name = name.upper()
        # Format common names for better readability
        formatting_map = {
            'SGD': 'SGD',
            'ADAM': 'Adam',
            'ADAMW': 'AdamW',
            'MUON': 'Muon',
            'EQPROP': 'EqProp',
            'SMEP': 'SMEP',
            'SDMEP': 'SDMEP',
            'LOCAL_EP': 'LocalEP',
            'NATURAL_EP': 'NaturalEP',
            'LOCALEPMUON': 'LocalEP',
            'NATURALEPMUON': 'NaturalEP',
        }
        return formatting_map.get(name, name)

    def plot_optimizer_comparison_with_stats(
        self,
        results: Dict[str, Any],
        metric: str = 'final_test_accuracy'
    ) -> None:
        """
        Compare optimizers with error bars and statistical significance.

        Args:
            results: Complete results dictionary from run_benchmarks
            metric: Metric to compare ('final_test_accuracy')
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        optimizer_names = []
        means = []
        stds = []

        for opt_name, opt_data in results['optimizers'].items():
            stats = opt_data['statistics']['final_test_accuracy']
            optimizer_names.append(self._format_optimizer_name(opt_name))
            means.append(stats['mean'] * 100)  # Convert to percentage
            stds.append(stats['std'] * 100)

        # Color scheme: distinguish baselines from EP variants
        colors = []
        for name in optimizer_names:
            name_lower = name.lower()
            if name_lower in ['sgd', 'adam', 'adamw']:
                colors.append('#F18F01')  # Orange for standard baselines
            elif name_lower == 'muon':
                colors.append('#A23B72')  # Purple for Muon (backprop)
            elif 'EP' in name or 'SDMEP' in name or 'SMEP' in name:
                colors.append('#2E86AB')  # Blue for EP variants
            else:
                colors.append('#666666')  # Gray for others

        x_pos = np.arange(len(optimizer_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        # Add significance markers
        baseline_idx = 0 if 'SGD' in optimizer_names else None
        if baseline_idx is not None:
            baseline_mean = means[baseline_idx]
            for i, (name, mean) in enumerate(zip(optimizer_names, means)):
                if i == baseline_idx:
                    continue
                opt_data = results['optimizers'].get(name.lower(), {})
                if 'comparison_to_baseline' in opt_data.get('statistics', {}):
                    comp = opt_data['statistics']['comparison_to_baseline']
                    if comp.get('significant', False):
                        # Add asterisk
                        y_pos = max(mean, baseline_mean) + 1
                        ax.text(i, y_pos, '*', ha='center', va='bottom',
                                fontsize=20, color='red', fontweight='bold')

        ax.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Optimizer Comparison: Final Test Accuracy\n(with 95% CI error bars)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizer_names, rotation=15, ha='right')
        ax.set_ylim(0, max(means) * 1.15)

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.5,
                    f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'optimizer_comparison_stats.png'),
                    bbox_inches='tight')
        plt.close()

    def plot_training_curves_all(
        self,
        results: Dict[str, Any]
    ) -> None:
        """
        Plot training curves for all optimizers across all repeats.

        Args:
            results: Complete results dictionary
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Color mapping for optimizers (consistent across all plots)
        color_map = {
            'sgd': '#F18F01',
            'adam': '#F18F01',
            'adamw': '#F18F01',
            'muon': '#A23B72',
            'eqprop': '#2E86AB',
            'smep': '#2E86AB',
            'sdmep': '#1a5f7a',
            'local_ep': '#5C946E',
            'natural_ep': '#3D5A80',
            'localepmuon': '#5C946E',
            'naturalepmuon': '#3D5A80',
        }

        # Plot training curves
        for opt_name, opt_data in results['optimizers'].items():
            color = color_map.get(opt_name.lower(), '#666666')
            for repeat_data in opt_data['repeats']:
                history = repeat_data['history']
                epochs = range(1, len(history['epoch_loss']) + 1)

                # Loss curve (left)
                if history['epoch_loss']:
                    axes[0].plot(epochs, history['epoch_loss'],
                                 color=color, alpha=0.6, linewidth=2.5,
                                 label=opt_name.upper() if repeat_data['repeat'] == 1 else None)

                # Accuracy curve (right)
                if history['epoch_accuracy']:
                    axes[1].plot(epochs, history['epoch_accuracy'],
                                 color=color, alpha=0.6, linewidth=2.5,
                                 label=opt_name.upper() if repeat_data['repeat'] == 1 else None)

        axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Training Loss', fontsize=11, fontweight='bold')
        axes[0].set_title('Training Loss Over Epochs\n(shaded: individual repeats)',
                          fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Training Accuracy', fontsize=11, fontweight='bold')
        axes[1].set_title('Training Accuracy Over Epochs\n(shaded: individual repeats)',
                          fontsize=12, fontweight='bold')
        axes[1].legend(loc='lower right', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves_all.png'),
                    bbox_inches='tight')
        plt.close()

    def plot_test_accuracy_comparison(
        self,
        results: Dict[str, Any]
    ) -> None:
        """
        Plot test accuracy over epochs for all optimizers.
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Consistent color mapping
        color_map = {
            'sgd': '#F18F01',
            'adam': '#F18F01',
            'adamw': '#F18F01',
            'muon': '#A23B72',
            'eqprop': '#2E86AB',
            'smep': '#2E86AB',
            'sdmep': '#1a5f7a',
            'local_ep': '#5C946E',
            'natural_ep': '#3D5A80',
            'localepmuon': '#5C946E',
            'naturalepmuon': '#3D5A80',
        }

        for opt_name, opt_data in results['optimizers'].items():
            color = color_map.get(opt_name.lower(), '#666666')
            for repeat_data in opt_data['repeats']:
                history = repeat_data['history']
                if history.get('test_accuracy'):
                    epochs = range(1, len(history['test_accuracy']) + 1)
                    ax.plot(epochs, np.array(history['test_accuracy']) * 100,
                            color=color, alpha=0.3, linewidth=1.5)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Test Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Create legend handles
        handles = []
        for opt_name in results['optimizers'].keys():
            color = color_map.get(opt_name.lower(), '#666666')
            handles.append(plt.Line2D([0], [0], color=color, linewidth=2,
                                       label=opt_name.upper()))

        ax.legend(handles=handles, loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'test_accuracy_comparison.png'),
                    bbox_inches='tight')
        plt.close()

    def plot_time_analysis(
        self,
        results: Dict[str, Any]
    ) -> None:
        """
        Plot time per epoch and time per step analysis.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Consistent color mapping
        color_map = {
            'sgd': '#F18F01',
            'adam': '#F18F01',
            'adamw': '#F18F01',
            'muon': '#A23B72',
            'eqprop': '#2E86AB',
            'smep': '#2E86AB',
            'sdmep': '#1a5f7a',
            'local_ep': '#5C946E',
            'natural_ep': '#3D5A80',
        }

        # 1. Time per Epoch
        optimizer_names = []
        epoch_means = []
        epoch_stds = []
        step_means = []
        step_stds = []

        for opt_name, opt_data in results['optimizers'].items():
            epoch_times = []
            step_times = []

            for repeat_data in opt_data['repeats']:
                history = repeat_data['history']
                if history.get('epoch_time'):
                    epoch_times.extend(history['epoch_time'])
                if history.get('time_per_step'):
                    # Flatten list of lists if needed, or handle list of step times
                    # Assuming time_per_step is list of mean times per epoch or similar
                    # Actually runner collects it per epoch? Let's check runner.
                    # We will modify runner to store mean time_per_step per epoch.
                    step_times.extend(history['time_per_step'])

            if epoch_times:
                optimizer_names.append(opt_name.upper())
                epoch_means.append(np.mean(epoch_times))
                epoch_stds.append(np.std(epoch_times))

                if step_times:
                    step_means.append(np.mean(step_times) * 1000) # Convert to ms
                    step_stds.append(np.std(step_times) * 1000)
                else:
                    step_means.append(0)
                    step_stds.append(0)

        if optimizer_names:
            x_pos = np.arange(len(optimizer_names))

            # Bar colors based on optimizer type
            bar_colors = [color_map.get(name.lower(), '#666666') for name in optimizer_names]

            # Epoch Time Plot
            bars1 = axes[0].bar(x_pos, epoch_means, yerr=epoch_stds, capsize=5,
                          color=bar_colors, alpha=0.8, edgecolor='black')
            axes[0].set_xlabel('Optimizer', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Time per Epoch (s)', fontsize=12, fontweight='bold')
            axes[0].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(optimizer_names, rotation=25, ha='right')

            for bar, mean, std in zip(bars1, epoch_means, epoch_stds):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width() / 2., height + std + 0.05,
                        f'{mean:.2f}s', ha='center', va='bottom', fontsize=9)

            # Step Time Plot
            bars2 = axes[1].bar(x_pos, step_means, yerr=step_stds, capsize=5,
                          color=bar_colors, alpha=0.8, edgecolor='black')
            axes[1].set_xlabel('Optimizer', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Time per Step (ms)', fontsize=12, fontweight='bold')
            axes[1].set_title('Computational Cost per Step', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(optimizer_names, rotation=25, ha='right')

            for bar, mean, std in zip(bars2, step_means, step_stds):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width() / 2., height + std + 0.1,
                        f'{mean:.1f}ms', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'time_analysis.png'),
                        bbox_inches='tight')
            plt.close()

    def generate_summary_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Generate a markdown summary report.

        Returns:
            Markdown formatted report string
        """
        report = []
        report.append("# MEP Benchmark Results\n")
        report.append(f"**Generated:** {results['metadata']['timestamp']}\n")
        report.append(f"**Device:** {results['metadata']['device']}\n")
        report.append(f"**Repeats:** {results['metadata']['repeats']}\n")
        report.append(f"**Target Time per Trial:** {results['metadata']['target_time_per_trial']}s\n")

        # Configuration
        config = results.get('config', {})
        report.append("\n## Configuration\n")
        report.append(f"- **Dataset:** {config.get('dataset', 'N/A')}")
        report.append(f"- **Model:** {config.get('model', 'N/A')}")
        if 'architecture' in config:
            dims = config['architecture'].get('dims', [])
            report.append(f"- **Architecture:** {' → '.join(map(str, dims))}")
        report.append(f"- **Batch Size:** {config.get('batch_size', 'N/A')}")
        report.append("")

        # Summary table
        report.append("\n## Results Summary\n")
        report.append("| Optimizer | Mean Acc (%) | Std (%) | Min (%) | Max (%) | vs SGD |")
        report.append("|-----------|--------------|---------|---------|---------|--------|")

        for opt_name, opt_data in results['optimizers'].items():
            stats = opt_data['statistics']['final_test_accuracy']
            mean_acc = stats['mean'] * 100
            std_acc = stats['std'] * 100
            min_acc = stats['min'] * 100
            max_acc = stats['max'] * 100

            comparison = "-"
            if 'comparison_to_baseline' in opt_data['statistics']:
                comp = opt_data['statistics']['comparison_to_baseline']
                if comp.get('significant', False):
                    comparison = "✓ Significant"
                else:
                    comparison = "No sig. diff"

            report.append(f"| {self._format_optimizer_name(opt_name)} | {mean_acc:.2f} | {std_acc:.2f} | {min_acc:.2f} | {max_acc:.2f} | {comparison} |")

        report.append("\n*Note: Statistical significance tested using Welch's t-test (α=0.05)*\n")

        return "\n".join(report)
