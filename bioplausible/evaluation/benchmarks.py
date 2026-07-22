"""
Pre-built benchmarks for standard tasks across all domains.

Each benchmark returns a callable that evaluates a model and returns BenchmarkResult.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch.nn as nn

from bioplausible.domains import DomainTask
from bioplausible.evaluation.base import (
    BenchmarkResult,
    MetricSuite,
    evaluate_model_on_task,
)

# ---------------------------------------------------------------------------
# Benchmark Registry
# ---------------------------------------------------------------------------


class BenchmarkRegistry:
    """Registry of named benchmarks."""

    _benchmarks: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, fn: Callable) -> None:
        cls._benchmarks[name] = fn

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._benchmarks:
            raise KeyError(
                f"Unknown benchmark: {name}. Available: {list(cls._benchmarks.keys())}"
            )
        return cls._benchmarks[name]

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._benchmarks.keys())


def get_benchmark(name: str) -> Callable:
    """Get a benchmark function by name."""
    return BenchmarkRegistry.get(name)


def list_benchmarks() -> List[str]:
    """List available benchmark names."""
    return BenchmarkRegistry.list()


# ---------------------------------------------------------------------------
# Domain-specific evaluator helper
# ---------------------------------------------------------------------------


def _make_benchmark(
    name: str,
    task_factory: Callable[[], DomainTask],
    metric_suite: MetricSuite,
    description: str = "",
    max_batches: Optional[int] = None,
) -> Callable:
    """Create a benchmark function from a task factory."""

    def benchmark_fn(model: nn.Module) -> BenchmarkResult:
        task = task_factory()
        task.setup()
        return evaluate_model_on_task(
            model,
            task,
            metric_suite=metric_suite,
            max_batches=max_batches,
        )

    benchmark_fn.__name__ = name
    benchmark_fn.__doc__ = description
    BenchmarkRegistry.register(name, benchmark_fn)
    return benchmark_fn


# ---------------------------------------------------------------------------
# Vision Benchmarks
# ---------------------------------------------------------------------------


def mnist_benchmark(model: nn.Module) -> BenchmarkResult:
    """Evaluate on MNIST (10-class digit classification)."""
    from bioplausible.domains import VisionTask

    task = VisionTask(name="mnist", dataset_name="mnist", batch_size=128)
    task.setup()
    return evaluate_model_on_task(
        model, task, metric_suite=MetricSuite.classification(), max_batches=100
    )


def cifar10_benchmark(model: nn.Module) -> BenchmarkResult:
    """Evaluate on CIFAR-10 (10-class object classification)."""
    from bioplausible.domains import VisionTask

    task = VisionTask(name="cifar10", dataset_name="cifar10", batch_size=128)
    task.setup()
    return evaluate_model_on_task(
        model, task, metric_suite=MetricSuite.multiclass(), max_batches=100
    )


def fashion_mnist_benchmark(model: nn.Module) -> BenchmarkResult:
    """Evaluate on Fashion-MNIST."""
    from bioplausible.domains import VisionTask

    task = VisionTask(
        name="fashion_mnist", dataset_name="fashion_mnist", batch_size=128
    )
    task.setup()
    return evaluate_model_on_task(
        model, task, metric_suite=MetricSuite.classification(), max_batches=100
    )


# ---------------------------------------------------------------------------
# Language Modeling Benchmarks
# ---------------------------------------------------------------------------


def tiny_shakespeare_benchmark(model: nn.Module) -> BenchmarkResult:
    """Evaluate on Tiny Shakespeare (character-level LM)."""
    from bioplausible.domains import LMTask

    task = LMTask(
        name="tiny_shakespeare",
        dataset_name="tiny_shakespeare",
        batch_size=32,
        seq_len=128,
    )
    task.setup()
    return evaluate_model_on_task(
        model, task, metric_suite=MetricSuite.language_modeling(), max_batches=50
    )


# ---------------------------------------------------------------------------
# Register all benchmarks
# ---------------------------------------------------------------------------

BenchmarkRegistry.register("mnist", mnist_benchmark)
BenchmarkRegistry.register("cifar10", cifar10_benchmark)
BenchmarkRegistry.register("fashion_mnist", fashion_mnist_benchmark)
BenchmarkRegistry.register("tiny_shakespeare", tiny_shakespeare_benchmark)

__all__ = [
    "BenchmarkRegistry",
    "get_benchmark",
    "list_benchmarks",
    "mnist_benchmark",
    "cifar10_benchmark",
    "fashion_mnist_benchmark",
    "tiny_shakespeare_benchmark",
]
