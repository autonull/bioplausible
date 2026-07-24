"""Evaluation package: standardized benchmarks and MetricSuite."""

from bioplausible.evaluation.base import BenchmarkResult
from bioplausible.evaluation.base import EvaluatorBase
from bioplausible.evaluation.base import MetricSuite
from bioplausible.evaluation.base import cross_validate
from bioplausible.evaluation.base import evaluate_model_on_task
from bioplausible.evaluation.base import registry_evaluator
from bioplausible.evaluation.benchmarks import BenchmarkRegistry
from bioplausible.evaluation.benchmarks import cifar10_benchmark
from bioplausible.evaluation.benchmarks import get_benchmark
from bioplausible.evaluation.benchmarks import list_benchmarks
from bioplausible.evaluation.benchmarks import mnist_benchmark
from bioplausible.evaluation.benchmarks import tiny_shakespeare_benchmark
from bioplausible.evaluation.cross_domain import BenchmarkSuiteConfig
from bioplausible.evaluation.cross_domain import BenchmarkSuiteResult
from bioplausible.evaluation.cross_domain import CrossDomainBenchmarkSuite
from bioplausible.evaluation.cross_domain import run_cross_domain_benchmark

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
    # Cross-domain suite
    "BenchmarkSuiteConfig",
    "BenchmarkSuiteResult",
    "CrossDomainBenchmarkSuite",
    "run_cross_domain_benchmark",
]
