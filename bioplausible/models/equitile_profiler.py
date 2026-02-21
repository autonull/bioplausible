"""
EquiTile Profiler: Performance Diagnostics
==========================================

Tools for profiling and diagnosing EquiTile performance:
- Tile-level timing
- Memory usage tracking
- Activity statistics
- Learning diagnostics

Usage:
    from bioplausible.models.equitile_profiler import EquiTileProfiler

    model = EquiTile(...)
    profiler = EquiTileProfiler(model)

    with profiler.profile():
        model.train_step(X, y)

    profiler.print_report()
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from .equitile import EquiTile


@dataclass
class TileStats:
    """Statistics for a single tile."""
    tile_id: int
    layer_id: int
    is_input: bool = False
    is_output: bool = False

    # Timing
    predict_time: float = 0.0
    update_time: float = 0.0
    total_time: float = 0.0

    # Activity
    activity_mean: float = 0.0
    activity_std: float = 0.0
    activity_max: float = 0.0
    error_norm: float = 0.0

    # Learning
    weight_update_norm: float = 0.0
    importance: float = 0.0

    # Computed fields
    call_count: int = 0


@dataclass
class ProfileResult:
    """Results from a profiling session."""
    # Tile-level stats
    tile_stats: Dict[int, TileStats] = field(default_factory=dict)

    # Aggregate stats
    total_time: float = 0.0
    predict_time: float = 0.0
    update_time: float = 0.0
    learning_time: float = 0.0

    # Memory
    param_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0

    # Metadata
    batch_size: int = 0
    n_tiles: int = 0
    n_edges: int = 0


class EquiTileProfiler:
    """Profiler for EquiTile models.

    Tracks timing, memory, and activity statistics.
    """

    def __init__(self, model: 'EquiTile'):
        self.model = model
        self._profiling = False
        self._current_result: Optional[ProfileResult] = None
        self._tile_timers: Dict[int, Dict[str, float]] = defaultdict(lambda: {
            'predict': 0.0,
            'update': 0.0,
        })
        self._section_timers: Dict[str, float] = defaultdict(float)
        self._history: List[ProfileResult] = []

    def profile(self) -> '_ProfileContext':
        """Context manager for profiling."""
        return _ProfileContext(self)

    def _start_profiling(self):
        """Start profiling."""
        self._profiling = True
        self._current_result = ProfileResult()
        self._tile_timers.clear()
        self._section_timers.clear()

    def _stop_profiling(self) -> ProfileResult:
        """Stop profiling and return results."""
        self._profiling = False

        # Aggregate tile stats
        result = self._current_result
        result.tile_stats = self._aggregate_tile_stats()
        result.total_time = sum(self._section_timers.values())
        result.predict_time = sum(t['predict'] for t in self._tile_timers.values())
        result.update_time = sum(t['update'] for t in self._tile_timers.values())
        result.n_tiles = len(self.model.graph.tiles)
        result.n_edges = len(self.model.graph.edges)

        # Memory stats
        result.param_memory_mb = self._measure_memory()
        result.activation_memory_mb = self._measure_activation_memory()

        # Store in history
        self._history.append(result)

        self._current_result = None
        return result

    def _aggregate_tile_stats(self) -> Dict[int, TileStats]:
        """Aggregate statistics per tile."""
        stats = {}

        for tile in self.model.graph.all_tiles:
            tile_idx = list(self.model.graph.tiles.keys()).index(tile.id)
            importance = torch.sigmoid(self.model.tile_importance[tile_idx]).item()

            activity_mean = 0.0
            activity_std = 0.0
            activity_max = 0.0
            error_norm = 0.0

            if tile.activity is not None:
                activity_mean = tile.activity.mean().item()
                activity_std = tile.activity.std().item()
                activity_max = tile.activity.abs().max().item()

            if tile.error is not None:
                error_norm = tile.error.norm(p=2).item()

            stats[tile.id] = TileStats(
                tile_id=tile.id,
                layer_id=tile.layer_id,
                is_input=tile.is_input,
                is_output=tile.is_output,
                predict_time=self._tile_timers[tile.id]['predict'],
                update_time=self._tile_timers[tile.id]['update'],
                total_time=self._tile_timers[tile.id]['predict'] + self._tile_timers[tile.id]['update'],
                activity_mean=activity_mean,
                activity_std=activity_std,
                activity_max=activity_max,
                error_norm=error_norm,
                importance=importance,
                call_count=1 if activity_mean != 0 else 0,
            )

        return stats

    def _measure_memory(self) -> float:
        """Measure parameter memory in MB."""
        param_mem = sum(p.numel() * p.element_size() for p in self.model.parameters())

        edge_mem = 0
        for edge in self.model.graph.edges.values():
            if edge.weight is not None:
                edge_mem += edge.weight.numel() * edge.weight.element_size()
            if edge.bias is not None:
                edge_mem += edge.bias.numel() * edge.bias.element_size()

        return (param_mem + edge_mem) / (1024 * 1024)

    def _measure_activation_memory(self) -> float:
        """Measure activation memory in MB."""
        activation_mem = 0

        for tile in self.model.graph.all_tiles:
            if tile.activity is not None:
                activation_mem += tile.activity.numel() * tile.activity.element_size()
            if tile.prediction is not None:
                activation_mem += tile.prediction.numel() * tile.prediction.element_size()
            if tile.error is not None:
                activation_mem += tile.error.numel() * tile.error.element_size()

        return activation_mem / (1024 * 1024)

    def time_predict(self, tile_id: int):
        """Context manager for timing prediction."""
        return _TimerContext(self._tile_timers[tile_id], 'predict')

    def time_update(self, tile_id: int):
        """Context manager for timing update."""
        return _TimerContext(self._tile_timers[tile_id], 'update')

    def time_section(self, section: str):
        """Context manager for timing a section."""
        return _TimerContext(self._section_timers, section)

    def print_report(self, last_n: int = 1):
        """Print profiling report."""
        if not self._history:
            print("No profiling data available.")
            return

        for result in self._history[-last_n:]:
            self._print_single_report(result)

    def _print_single_report(self, result: ProfileResult):
        """Print a single profiling report."""
        print()
        print("=" * 70)
        print("EquiTile Profiling Report")
        print("=" * 70)
        print()

        # Summary
        total_time = result.total_time if result.total_time > 0 else 0.001
        predict_pct = (result.predict_time / total_time * 100) if total_time > 0 else 0
        update_pct = (result.update_time / total_time * 100) if total_time > 0 else 0

        print("Summary:")
        print(f"  Total time: {total_time*1000:.2f} ms")
        print(f"  Predict time: {result.predict_time*1000:.2f} ms ({predict_pct:.1f}%)")
        print(f"  Update time: {result.update_time*1000:.2f} ms ({update_pct:.1f}%)")
        print(f"  Batch size: {result.batch_size}")
        print(f"  Tiles: {result.n_tiles}, Edges: {result.n_edges}")
        print()

        # Memory
        print("Memory:")
        print(f"  Parameters: {result.param_memory_mb:.2f} MB")
        print(f"  Activations: {result.activation_memory_mb:.2f} MB")
        print(f"  Total: {result.param_memory_mb + result.activation_memory_mb:.2f} MB")
        print()

        # Tile breakdown
        print("Tile Breakdown (top 5 by time):")
        sorted_tiles = sorted(
            result.tile_stats.values(),
            key=lambda s: s.total_time,
            reverse=True
        )[:5]

        print(f"  {'ID':>4} {'Layer':>6} {'Time(ms)':>10} {'Error':>10} {'Importance':>10}")
        print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

        for tile in sorted_tiles:
            print(f"  {tile.tile_id:>4} {tile.layer_id:>6} "
                  f"{tile.total_time*1000:>10.2f} {tile.error_norm:>10.2f} "
                  f"{tile.importance:>10.3f}")

        print()

        # Activity stats
        print("Activity Statistics:")
        activities = [s.activity_mean for s in result.tile_stats.values() if not s.is_input]
        errors = [s.error_norm for s in result.tile_stats.values() if not s.is_input]

        if activities:
            print(f"  Activity mean: {sum(activities)/len(activities):.4f}")
            print(f"  Activity max: {max(s.activity_max for s in result.tile_stats.values()):.4f}")
        if errors:
            print(f"  Error mean: {sum(errors)/len(errors):.4f}")
            print(f"  Error max: {max(errors):.4f}")

        print()
        print("=" * 70)


class _ProfileContext:
    """Context manager for profiling."""

    def __init__(self, profiler: EquiTileProfiler):
        self.profiler = profiler

    def __enter__(self):
        self.profiler._start_profiling()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler._stop_profiling()
        return False


class _TimerContext:
    """Context manager for timing."""

    def __init__(self, timer_dict: Dict[str, float], key: str):
        self.timer_dict = timer_dict
        self.key = key
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            self.timer_dict[self.key] += elapsed
        return False


class LearningMonitor:
    """Monitors learning progress over time."""

    def __init__(self, model: 'EquiTile', window_size: int = 100):
        self.model = model
        self.window_size = window_size

        self._loss_history: List[float] = []
        self._accuracy_history: List[float] = []
        self._error_history: Dict[int, List[float]] = defaultdict(list)
        self._importance_history: List[float] = []

    def record(self, stats: Dict[str, float]):
        """Record training statistics."""
        self._loss_history.append(stats.get('loss', 0.0))
        self._accuracy_history.append(stats.get('accuracy', 0.0))
        self._importance_history.append(
            torch.sigmoid(self.model.tile_importance).mean().item()
        )

        # Per-tile errors
        for tile in self.model.graph.all_tiles:
            if tile.error is not None:
                error_norm = tile.error.norm(p=2).item()
                self._error_history[tile.id].append(error_norm)

        # Trim history
        if len(self._loss_history) > self.window_size:
            self._loss_history.pop(0)
            self._accuracy_history.pop(0)
            self._importance_history.pop(0)

        for tile_id in self._error_history:
            if len(self._error_history[tile_id]) > self.window_size:
                self._error_history[tile_id].pop(0)

    def get_summary(self) -> Dict[str, Any]:
        """Get learning summary."""
        if not self._loss_history:
            return {}

        return {
            'loss_mean': sum(self._loss_history[-10:]) / min(10, len(self._loss_history)),
            'loss_trend': self._compute_trend(self._loss_history),
            'accuracy_mean': sum(self._accuracy_history[-10:]) / min(10, len(self._accuracy_history)),
            'accuracy_trend': self._compute_trend(self._accuracy_history),
            'importance_mean': sum(self._importance_history[-10:]) / min(10, len(self._importance_history)),
            'hot_tiles': self._get_hot_tiles(),
        }

    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction."""
        if len(values) < 5:
            return 'stable'

        recent = sum(values[-5:]) / 5
        older = sum(values[-10:-5]) / 5

        if recent < older * 0.95:
            return 'decreasing'
        elif recent > older * 1.05:
            return 'increasing'
        return 'stable'

    def _get_hot_tiles(self) -> List[int]:
        """Get tiles with highest recent error."""
        if not self._error_history:
            return []

        avg_errors = {
            tile_id: sum(errors[-10:]) / min(10, len(errors))
            for tile_id, errors in self._error_history.items()
        }

        sorted_tiles = sorted(avg_errors.items(), key=lambda x: x[1], reverse=True)
        return [tile_id for tile_id, _ in sorted_tiles[:5]]

    def print_status(self):
        """Print current learning status."""
        summary = self.get_summary()

        if not summary:
            print("No data recorded yet.")
            return

        print()
        print("Learning Status:")
        print(f"  Loss: {summary['loss_mean']:.4f} ({summary['loss_trend']})")
        print(f"  Accuracy: {summary['accuracy_mean']:.4f} ({summary['accuracy_trend']})")
        print(f"  Mean Importance: {summary['importance_mean']:.4f}")

        if summary['hot_tiles']:
            print(f"  Hot Tiles: {summary['hot_tiles']}")

        print()
