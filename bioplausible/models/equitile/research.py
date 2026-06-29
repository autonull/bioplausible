"""
EquiTile Research Utilities
============================

Tools for facilitating research experiments:
- Experiment tracking hooks
- Metric collection APIs
- Visualization helpers
- Ablation study support

Examples
--------
>>> from bioplausible.models.equitile.research import ExperimentTracker
>>> tracker = ExperimentTracker(experiment_name="my_experiment")
>>> tracker.log_params({"learning_rate": 0.01, "batch_size": 32})
>>> for epoch in range(100):
...     stats = model.train_step(X, y)
...     tracker.log_metrics(stats, epoch=epoch)
>>> tracker.save()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from .core import EquiTile


# =============================================================================
# Experiment Tracker
# =============================================================================


@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Attributes
    ----------
    name : str
        Experiment name
    description : str
        Experiment description
    tags : list of str
        Experiment tags
    """

    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)


class ExperimentTracker:
    """Tracks experiment parameters, metrics, and artifacts.

    Parameters
    ----------
    experiment_name : str
        Experiment name
    log_dir : str, optional
        Directory for logs
    config : ExperimentConfig, optional
        Experiment configuration

    Examples
    --------
    >>> tracker = ExperimentTracker("mnist_experiment")
    >>> tracker.log_params({"lr": 0.01, "batch_size": 32})
    >>> tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=100)
    >>> tracker.save()
    """

    def __init__(
        self,
        experiment_name: str = "",
        log_dir: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
    ) -> None:
        self.config = config or ExperimentConfig(name=experiment_name)
        self.experiment_name = experiment_name or self.config.name

        # Set up log directory
        if log_dir is None:
            log_dir = os.path.join("logs", "equitile", self.experiment_name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tracking state
        self._params: Dict[str, Any] = {}
        self._metrics: List[Dict[str, Any]] = []
        self._artifacts: List[str] = []
        self._start_time = time.time()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters.

        Parameters
        ----------
        params : dict
            Parameters to log
        """
        self._params.update(params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics.

        Parameters
        ----------
        metrics : dict
            Metrics to log
        step : int, optional
            Training step
        epoch : int, optional
            Epoch number
        """
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "step": step,
            "epoch": epoch,
            **metrics,
        }
        self._metrics.append(entry)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log an artifact (file).

        Parameters
        ----------
        path : str
            Path to artifact
        name : str, optional
            Artifact name
        """
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        # Copy to log directory
        artifact_name = name or artifact_path.name
        dest_path = self.log_dir / artifact_name

        # Read and write to preserve
        with open(artifact_path, "rb") as f:
            content = f.read()
        with open(dest_path, "wb") as f:
            f.write(content)

        self._artifacts.append(str(dest_path))

    def log_model(
        self,
        model: EquiTile,
        name: str = "model",
        include_graph: bool = False,
    ) -> None:
        """Log model checkpoint.

        Parameters
        ----------
        model : EquiTile
            Model to save
        name : str
            Model name
        include_graph : bool
            Include model graph
        """
        path = self.log_dir / f"{name}.pt"
        model.save_checkpoint(str(path))
        self.log_artifact(str(path))

        if include_graph:
            graph_path = self.log_dir / f"{name}_graph.json"
            self._save_model_graph(model, str(graph_path))
            self.log_artifact(str(graph_path))

    def _save_model_graph(self, model: EquiTile, path: str) -> None:
        """Save model graph to JSON.

        Parameters
        ----------
        model : EquiTile
            Model
        path : str
            Output path
        """
        graph_data = {
            "n_tiles": len(model.graph.tiles),
            "n_edges": len(model.graph.edges),
            "tiles": [
                {
                    "id": tile.id,
                    "layer": tile.layer_id,
                    "neurons": tile.neurons,
                    "is_input": tile.is_input,
                    "is_output": tile.is_output,
                }
                for tile in model.graph.all_tiles
            ],
        }
        with open(path, "w") as f:
            json.dump(graph_data, f, indent=2)

    def get_metrics(
        self,
        metric_name: str,
        as_array: bool = True,
    ) -> Union[List[float], List[Dict[str, Any]]]:
        """Get logged metrics.

        Parameters
        ----------
        metric_name : str
            Metric name
        as_array : bool
            Return as array of values

        Returns
        -------
        list
            Metrics
        """
        if as_array:
            return [m.get(metric_name) for m in self._metrics if metric_name in m]
        return [m for m in self._metrics if metric_name in m]

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self._metrics:
            return {}

        # Compute summary statistics for numeric metrics
        summary: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "n_steps": len(self._metrics),
            "duration_seconds": time.time() - self._start_time,
            "params": self._params,
        }

        # Get all metric keys
        metric_keys = set()
        for m in self._metrics:
            metric_keys.update(
                k for k in m.keys() if k not in ("timestamp", "step", "epoch")
            )

        # Compute stats for each metric
        for key in metric_keys:
            values = self.get_metrics(key)
            if values and all(v is not None for v in values):
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_final"] = values[-1]

        return summary

    def save(self) -> str:
        """Save experiment data.

        Returns
        -------
        str
            Path to saved file
        """
        # Save params
        params_path = self.log_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(self._params, f, indent=2)

        # Save metrics
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2)

        # Save summary
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        return str(self.log_dir)

    def export_csv(self, path: Optional[str] = None) -> str:
        """Export metrics to CSV.

        Parameters
        ----------
        path : str, optional
            Output path

        Returns
        -------
        str
            Path to CSV file
        """
        if path is None:
            path = str(self.log_dir / "metrics.csv")

        if not self._metrics:
            return path

        # Get all keys
        keys = set()
        for m in self._metrics:
            keys.update(m.keys())

        # Write CSV
        with open(path, "w") as f:
            # Header
            f.write(",".join(sorted(keys)) + "\n")

            # Rows
            for m in self._metrics:
                values = [str(m.get(k, "")) for k in sorted(keys)]
                f.write(",".join(values) + "\n")

        return path


# =============================================================================
# Metric Collector
# =============================================================================


@dataclass
class MetricEntry:
    """Single metric entry.

    Attributes
    ----------
    name : str
        Metric name
    value : float
        Metric value
    step : int
            Training step
    timestamp : float
        Unix timestamp
    tags : dict
        Additional tags
    """

    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Collects and aggregates metrics.

    Parameters
    ----------
    window_size : int
        Window size for moving averages
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._metrics: Dict[str, List[MetricEntry]] = {}
        self._step = 0

    def add(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a metric.

        Parameters
        ----------
        name : str
            Metric name
        value : float
            Metric value
        tags : dict, optional
            Additional tags
        """
        if name not in self._metrics:
            self._metrics[name] = []

        entry = MetricEntry(
            name=name,
            value=value,
            step=self._step,
            tags=tags or {},
        )
        self._metrics[name].append(entry)

        # Trim history
        if len(self._metrics[name]) > self.window_size:
            self._metrics[name].pop(0)

    def step(self) -> None:
        """Increment step counter."""
        self._step += 1

    def get(self, name: str) -> List[float]:
        """Get metric values.

        Parameters
        ----------
        name : str
            Metric name

        Returns
        -------
        list
            Metric values
        """
        if name not in self._metrics:
            return []
        return [e.value for e in self._metrics[name]]

    def get_mean(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """Get mean of metric.

        Parameters
        ----------
        name : str
            Metric name
        window : int, optional
            Window size

        Returns
        -------
        float, optional
            Mean value
        """
        values = self.get(name)
        if not values:
            return None

        if window is not None:
            values = values[-window:]

        return sum(values) / len(values)

    def get_trend(self, name: str, window: int = 10) -> str:
        """Get metric trend.

        Parameters
        ----------
        name : str
            Metric name
        window : int
            Window size

        Returns
        -------
        str
            Trend direction
        """
        values = self.get(name)
        if len(values) < window * 2:
            return "stable"

        recent = sum(values[-window:]) / window
        older = sum(values[-window * 2 : -window]) / window

        if recent < older * 0.95:
            return "decreasing"
        elif recent > older * 1.05:
            return "increasing"
        return "stable"

    def get_all(self) -> Dict[str, List[float]]:
        """Get all metrics.

        Returns
        -------
        dict
            All metrics
        """
        return {name: self.get(name) for name in self._metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._step = 0


# =============================================================================
# Visualization Helpers
# =============================================================================


class VisualizationHelper:
    """Visualization helpers for EquiTile.

    Parameters
    ----------
    model : EquiTile
        Model to visualize
    """

    def __init__(self, model: EquiTile) -> None:
        self.model = model

    def get_tile_activities(self) -> Dict[int, torch.Tensor]:
        """Get tile activities.

        Returns
        -------
        dict
            Activities per tile
        """
        return {
            tile.id: tile.activity
            for tile in self.model.graph.all_tiles
            if tile.activity is not None
        }

    def get_tile_errors(self) -> Dict[int, torch.Tensor]:
        """Get tile errors.

        Returns
        -------
        dict
            Errors per tile
        """
        return {
            tile.id: tile.error
            for tile in self.model.graph.all_tiles
            if tile.error is not None
        }

    def get_importance_map(self) -> Dict[int, float]:
        """Get tile importance map.

        Returns
        -------
        dict
            Importance per tile
        """
        return {
            tile.id: torch.sigmoid(self.model.tile_importance[i]).item()
            for i, tile in enumerate(self.model.graph.all_tiles)
        }

    def get_error_heatmap_data(self) -> List[List[float]]:
        """Get error data for heatmap visualization.

        Returns
        -------
        list
            2D error array
        """
        # Organize by layer
        layers: Dict[int, List[float]] = {}
        for tile in self.model.graph.all_tiles:
            layer = tile.layer_id
            if layer not in layers:
                layers[layer] = []

            if tile.error is not None:
                error_norm = tile.error.norm(p=2).item()
            else:
                error_norm = 0.0

            layers[layer].append(error_norm)

        # Convert to 2D array
        max_tiles = max(len(tiles) for tiles in layers.values()) if layers else 1
        heatmap = []
        for layer_id in sorted(layers.keys()):
            row = layers[layer_id] + [0.0] * (max_tiles - len(layers[layer_id]))
            heatmap.append(row)

        return heatmap

    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data for visualization.

        Returns
        -------
        dict
            Graph data
        """
        nodes = []
        edges = []

        for tile in self.model.graph.all_tiles:
            nodes.append(
                {
                    "id": tile.id,
                    "layer": tile.layer_id,
                    "neurons": tile.neurons,
                    "is_input": tile.is_input,
                    "is_output": tile.is_output,
                    "pos_x": tile.pos_x,
                    "pos_y": tile.pos_y,
                }
            )

        for (src, dst), edge in self.model.graph.edges.items():
            edges.append(
                {
                    "source": src,
                    "target": dst,
                    "weight_norm": (
                        edge.weight.norm().item() if edge.weight is not None else 0.0
                    ),
                }
            )

        return {"nodes": nodes, "edges": edges}

    def plot_activities(self, ax=None):
        """Plot tile activities.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        matplotlib.axes.Axes
            Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")

        if ax is None:
            fig, ax = plt.subplots()

        activities = self.get_tile_activities()

        tile_ids = []
        means = []
        stds = []

        for tile_id, activity in activities.items():
            tile_ids.append(tile_id)
            means.append(activity.mean().item())
            stds.append(activity.std().item())

        ax.bar(range(len(tile_ids)), means, yerr=stds, capsize=3)
        ax.set_xlabel("Tile ID")
        ax.set_ylabel("Activity")
        ax.set_title("Tile Activities")

        return ax

    def plot_errors(self, ax=None):
        """Plot tile errors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        matplotlib.axes.Axes
            Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization")

        if ax is None:
            fig, ax = plt.subplots()

        errors = self.get_tile_errors()

        tile_ids = []
        norms = []

        for tile_id, error in errors.items():
            tile_ids.append(tile_id)
            norms.append(error.norm().item())

        ax.bar(range(len(tile_ids)), norms)
        ax.set_xlabel("Tile ID")
        ax.set_ylabel("Error Norm")
        ax.set_title("Tile Errors")

        return ax


# =============================================================================
# Ablation Study Support
# =============================================================================


@dataclass
class AblationConfig:
    """Ablation study configuration.

    Attributes
    ----------
    name : str
        Study name
    baseline_params : dict
        Baseline parameters
    variants : list
            List of parameter variants
    """

    name: str
    baseline_params: Dict[str, Any]
    variants: List[Dict[str, Any]]


class AblationStudy:
    """Support for ablation studies.

    Parameters
    ----------
    config : AblationConfig
        Study configuration
    log_dir : str, optional
        Log directory
    """

    def __init__(
        self,
        config: AblationConfig,
        log_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.log_dir = Path(log_dir or os.path.join("logs", "ablation", config.name))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._results: Dict[str, Dict[str, Any]] = {}

    def run_variant(
        self,
        variant_id: str,
        variant_params: Dict[str, Any],
        train_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run a single variant.

        Parameters
        ----------
        variant_id : str
            Variant identifier
        variant_params : dict
            Variant parameters
        train_fn : callable
            Training function

        Returns
        -------
        dict
            Results
        """
        # Merge with baseline
        params = {**self.config.baseline_params, **variant_params}

        # Create tracker
        tracker = ExperimentTracker(
            experiment_name=f"{self.config.name}_{variant_id}",
            log_dir=str(self.log_dir / variant_id),
        )
        tracker.log_params(params)

        # Run training
        results = train_fn(params)

        # Log results
        tracker.log_metrics(results)
        tracker.save()

        self._results[variant_id] = results
        return results

    def run_all(
        self,
        train_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Run all variants.

        Parameters
        ----------
        train_fn : callable
            Training function

        Returns
        -------
        dict
            All results
        """
        # Run baseline
        baseline_id = "baseline"
        self._results[baseline_id] = self.run_variant(
            baseline_id,
            {},
            train_fn,
        )

        # Run variants
        for i, variant in enumerate(self.config.variants):
            variant_id = f"variant_{i}"
            self._results[variant_id] = self.run_variant(
                variant_id,
                variant,
                train_fn,
            )

        return self._results

    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison of all variants.

        Returns
        -------
        dict
            Comparison data
        """
        comparison = {
            "study_name": self.config.name,
            "variants": list(self._results.keys()),
            "results": self._results,
        }

        # Save comparison
        path = self.log_dir / "comparison.json"
        with open(path, "w") as f:
            json.dump(comparison, f, indent=2)

        return comparison

    def export_table(self) -> str:
        """Export results as markdown table.

        Returns
        -------
        str
            Markdown table
        """
        if not self._results:
            return ""

        # Get all metric keys
        all_keys = set()
        for results in self._results.values():
            all_keys.update(results.keys())

        # Build table
        lines = []
        lines.append("| Variant | " + " | ".join(sorted(all_keys)) + " |")
        lines.append("|" + "|".join(["---"] * (len(all_keys) + 1)) + "|")

        for variant_id, results in self._results.items():
            values = [str(results.get(k, "N/A")) for k in sorted(all_keys)]
            lines.append(f"| {variant_id} | " + " | ".join(values) + " |")

        table = "\n".join(lines)

        # Save
        path = self.log_dir / "results.md"
        with open(path, "w") as f:
            f.write(table)

        return table


# =============================================================================
# Factory Functions
# =============================================================================


def create_tracker(
    experiment_name: str,
    log_dir: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> ExperimentTracker:
    """Create an experiment tracker.

    Parameters
    ----------
    experiment_name : str
        Experiment name
    log_dir : str, optional
        Log directory
    tags : list, optional
        Experiment tags

    Returns
    -------
    ExperimentTracker
        Tracker
    """
    config = ExperimentConfig(
        name=experiment_name,
        tags=tags or [],
    )
    return ExperimentTracker(experiment_name, log_dir, config)


def create_metric_collector(window_size: int = 100) -> MetricCollector:
    """Create a metric collector.

    Parameters
    ----------
    window_size : int
        Window size

    Returns
    -------
    MetricCollector
        Collector
    """
    return MetricCollector(window_size)


def create_visualization_helper(model: EquiTile) -> VisualizationHelper:
    """Create a visualization helper.

    Parameters
    ----------
    model : EquiTile
        Model

    Returns
    -------
    VisualizationHelper
        Helper
    """
    return VisualizationHelper(model)


def create_ablation_study(
    name: str,
    baseline_params: Dict[str, Any],
    variants: List[Dict[str, Any]],
    log_dir: Optional[str] = None,
) -> AblationStudy:
    """Create an ablation study.

    Parameters
    ----------
    name : str
        Study name
    baseline_params : dict
        Baseline parameters
    variants : list
        Variants
    log_dir : str, optional
        Log directory

    Returns
    -------
    AblationStudy
        Study
    """
    config = AblationConfig(
        name=name,
        baseline_params=baseline_params,
        variants=variants,
    )
    return AblationStudy(config, log_dir)
