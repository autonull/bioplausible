"""Tests for the evaluation package."""

import pytest
import torch
from torch import nn

from bioplausible.evaluation.base import (
    BenchmarkResult,
    MetricSuite,
    accuracy_fn,
    evaluate_model_on_task,
    mse_fn,
    perplexity_fn,
)
from bioplausible.evaluation.benchmarks import get_benchmark, list_benchmarks


class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)


class TestMetricSuite:
    def test_classification_defaults(self):
        suite = MetricSuite.classification()
        assert len(suite.metrics) >= 1
        assert suite.metrics[0].name == "accuracy"

    def test_lm_defaults(self):
        suite = MetricSuite.language_modeling()
        names = [m.name for m in suite.metrics]
        assert "accuracy" in names
        assert "perplexity" in names

    def test_regression_defaults(self):
        suite = MetricSuite.regression()
        names = [m.name for m in suite.metrics]
        assert "mse" in names
        assert "mae" in names

    def test_custom_metrics(self):
        suite = MetricSuite.custom(["accuracy", "f1", "mse"])
        assert len(suite.metrics) == 3

    def test_custom_unknown_metric(self):
        suite = MetricSuite.custom(["nonexistent"])
        assert len(suite.metrics) == 0

    def test_evaluate(self):
        suite = MetricSuite.classification()
        outputs = torch.tensor([[2.0, 1.0, 0.1], [0.1, 3.0, 0.2]])
        targets = torch.tensor([2, 1])
        results = suite.evaluate(outputs, targets)
        assert "accuracy" in results

    def test_best_direction(self):
        suite = MetricSuite.classification()
        assert suite.best_direction("accuracy") == "maximize"


class TestMetricFn:
    def test_accuracy(self):
        outputs = torch.tensor([[2.0, 1.0], [0.1, 2.0], [3.0, 0.1]])
        targets = torch.tensor([0, 1, 2])
        acc = accuracy_fn(outputs, targets)
        assert acc == pytest.approx(2.0 / 3.0)

    def test_perplexity(self):
        outputs = torch.tensor([[2.0, 1.0], [0.5, 3.0]])
        targets = torch.tensor([0, 1])
        ppl = perplexity_fn(outputs, targets)
        assert ppl > 0

    def test_mse(self):
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, 2.0], [3.5, 4.5]])
        loss = mse_fn(outputs, targets)
        assert loss > 0


class TestBenchmarkResult:
    def test_create(self):
        result = BenchmarkResult(
            model_name="test",
            task_name="mnist",
            metrics={"accuracy": 0.95, "loss": 0.1},
            params_count=1000,
        )
        assert result.model_name == "test"
        assert result.metrics["accuracy"] == 0.95

    def test_summary(self):
        result = BenchmarkResult(
            model_name="MLP",
            task_name="mnist",
            metrics={"accuracy": 0.95},
        )
        summary = result.summary()
        assert "MLP" in summary
        assert "mnist" in summary

    def test_to_dict(self):
        result = BenchmarkResult(
            model_name="MLP",
            task_name="mnist",
            metrics={"accuracy": 0.95},
        )
        d = result.to_dict()
        assert d["model_name"] == "MLP"
        assert d["metrics"]["accuracy"] == 0.95


class TestBenchmarkRegistry:
    def test_list_benchmarks(self):
        benchmarks = list_benchmarks()
        assert "mnist" in benchmarks
        assert "cifar10" in benchmarks

    def test_get_benchmark(self):
        fn = get_benchmark("mnist")
        assert callable(fn)

    def test_get_unknown_benchmark(self):
        with pytest.raises(KeyError):
            get_benchmark("nonexistent")


class TestEvaluateModelOnTask:
    def test_with_vision_task(self):
        model = SimpleModel(784, 10)
        from bioplausible.domains import VisionTask

        task = VisionTask(
            name="test_mnist",
            dataset_name="mnist",
            batch_size=64,
        )
        task.setup()

        result = evaluate_model_on_task(model, task, max_batches=2)
        assert result.model_name == "SimpleModel"
        assert "accuracy" in result.metrics
        assert result.params_count > 0
