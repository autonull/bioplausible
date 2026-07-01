"""Evaluation package: standardized benchmarks and MetricSuite."""

from bioplausible.evaluation.base import (BenchmarkResult, EvaluatorBase,
                                          MetricSuite, cross_validate,
                                          evaluate_model_on_task,
                                          registry_evaluator)
from bioplausible.evaluation.benchmarks import (BenchmarkRegistry,
                                                cifar10_benchmark,
                                                get_benchmark, list_benchmarks,
                                                mnist_benchmark,
                                                tiny_shakespeare_benchmark)

__all__ = [
    # Base
    "EvaluatorBase",
    "MetricSuite",
    "BenchmarkResult",
    "evaluate_model_on_task",
    "cross_validate",
    "registry_evaluator",
    # Benchmarks
    "BenchmarkRegistry",
    "get_benchmark",
    "list_benchmarks",
    "mnist_benchmark",
    "cifar10_benchmark",
    "tiny_shakespeare_benchmark",
]
