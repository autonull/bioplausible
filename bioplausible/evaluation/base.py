"""
EvaluatorBase and MetricSuite: standardized evaluation framework.

Provides:
- MetricSuite: composable collection of metrics with domain-specific defaults
- EvaluatorBase: abstract evaluator for standardized model evaluation
- evaluate_model_on_task: convenience function
- registry_evaluator: decorator for registering evaluators
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

import torch
import torch.nn as nn

from bioplausible.domains.base import DomainTask
from bioplausible.domains.base import TaskSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric Suite
# ---------------------------------------------------------------------------


class MetricFn:
    """Wraps a metric computation function."""

    def __init__(
        self,
        name: str,
        fn: Callable[[torch.Tensor, torch.Tensor], float],
        higher_is_better: bool = True,
    ):
        self.name = name
        self.fn = fn
        self.higher_is_better = higher_is_better

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return self.fn(outputs, targets)


# Standard metric functions


def accuracy_fn(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Standard accuracy metric."""
    return (outputs.argmax(1) == targets).float().mean().item()


def top5_accuracy_fn(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-5 accuracy."""
    _, top5 = outputs.topk(5, dim=1)
    return top5.eq(targets.view(-1, 1)).any(dim=1).float().mean().item()


def perplexity_fn(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Perplexity from logits."""
    import numpy as np

    loss = nn.functional.cross_entropy(outputs, targets).item()
    return float(np.exp(min(loss, 10)))


def mse_fn(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean squared error."""
    return nn.functional.mse_loss(outputs, targets.float()).item()


def mae_fn(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error."""
    return nn.functional.l1_loss(outputs, targets.float()).item()


def f1_fn(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """F1 score for multi-class classification."""
    from sklearn.metrics import f1_score

    preds = outputs.argmax(1).cpu().numpy()
    return float(f1_score(targets.cpu().numpy(), preds, average="macro"))


# Pre-built metric suites

_CLASSIFICATION_METRICS = [
    MetricFn("accuracy", accuracy_fn, higher_is_better=True),
]

_MULTICLASS_METRICS = [
    MetricFn("accuracy", accuracy_fn, higher_is_better=True),
    MetricFn("top5_accuracy", top5_accuracy_fn, higher_is_better=True),
]

_LM_METRICS = [
    MetricFn("accuracy", accuracy_fn, higher_is_better=True),
    MetricFn("perplexity", perplexity_fn, higher_is_better=False),
]

_REGRESSION_METRICS = [
    MetricFn("mse", mse_fn, higher_is_better=False),
    MetricFn("mae", mae_fn, higher_is_better=False),
]


@dataclass
class MetricSuite:
    """
    Composable collection of metrics.

    Usage:
        suite = MetricSuite.classification()
        suite = MetricSuite.language_modeling()
        suite = MetricSuite(["accuracy", "f1"])
    """

    metrics: List[MetricFn] = field(default_factory=list)

    @classmethod
    def classification(cls) -> MetricSuite:
        """Standard classification metrics."""
        return cls(_CLASSIFICATION_METRICS)

    @classmethod
    def multiclass(cls) -> MetricSuite:
        """Multi-class classification with top-5."""
        return cls(_MULTICLASS_METRICS)

    @classmethod
    def language_modeling(cls) -> MetricSuite:
        """Language modeling metrics (accuracy + perplexity)."""
        return cls(_LM_METRICS)

    @classmethod
    def regression(cls) -> MetricSuite:
        """Regression metrics."""
        return cls(_REGRESSION_METRICS)

    @classmethod
    def custom(cls, metric_names: List[str]) -> MetricSuite:
        """Build a suite from standard metric names."""
        registry = {
            "accuracy": MetricFn("accuracy", accuracy_fn),
            "top5_accuracy": MetricFn("top5_accuracy", top5_accuracy_fn),
            "perplexity": MetricFn("perplexity", perplexity_fn, higher_is_better=False),
            "mse": MetricFn("mse", mse_fn, higher_is_better=False),
            "mae": MetricFn("mae", mae_fn, higher_is_better=False),
            "f1": MetricFn("f1", f1_fn),
        }
        metrics = []
        for name in metric_names:
            if name in registry:
                metrics.append(registry[name])
            else:
                logger.warning(f"Unknown metric: {name}")
        return cls(metrics)

    def evaluate(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate all metrics on outputs and targets."""
        return {m.name: m(outputs, targets) for m in self.metrics}

    def best_direction(self, metric_name: str = "accuracy") -> str:
        """Return 'maximize' or 'minimize' for a metric."""
        for m in self.metrics:
            if m.name == metric_name:
                return "maximize" if m.higher_is_better else "minimize"
        return "maximize"


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""

    model_name: str
    task_name: str
    metrics: Dict[str, float]
    params_count: Optional[int] = None
    flops: Optional[Dict[str, int]] = None
    energy_proxy: Optional[float] = None
    wall_time_s: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [f"{self.model_name} on {self.task_name}:"]
        for k, v in self.metrics.items():
            parts.append(f"  {k}: {v:.4f}")
        if self.params_count:
            parts.append(f"  params: {self.params_count:,}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "metrics": self.metrics,
            "params_count": self.params_count,
            "flops": self.flops,
            "energy_proxy": self.energy_proxy,
            "wall_time_s": self.wall_time_s,
            "peak_memory_mb": self.peak_memory_mb,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Evaluator Base
# ---------------------------------------------------------------------------


class EvaluatorBase(ABC):
    """
    Base class for domain-specific evaluators.

    Subclasses implement evaluate_model() for a specific domain/task.
    """

    def __init__(
        self,
        task: DomainTask,
        metric_suite: Optional[MetricSuite] = None,
    ):
        self.task = task
        self.metric_suite = metric_suite or MetricSuite.classification()

    @abstractmethod
    def evaluate_model(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: Optional[int] = None,
    ) -> BenchmarkResult:
        """Evaluate a model and return results."""
        pass

    def compare(
        self,
        models: Dict[str, nn.Module],
        split: TaskSplit = TaskSplit.VAL,
        max_batches: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple models on the same task."""
        results = {}
        for name, model in models.items():
            results[name] = self.evaluate_model(model, split, max_batches)
        return results


# ---------------------------------------------------------------------------
# Registry for evaluators
# ---------------------------------------------------------------------------

_EVALUATOR_REGISTRY: Dict[str, Type[EvaluatorBase]] = {}


def registry_evaluator(name: str) -> Callable:
    """Decorator to register an evaluator class."""

    def decorator(cls: Type[EvaluatorBase]) -> Type[EvaluatorBase]:
        _EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Convenience: evaluate_model_on_task
# ---------------------------------------------------------------------------


def evaluate_model_on_task(
    model: nn.Module,
    task: DomainTask,
    metric_suite: Optional[MetricSuite] = None,
    split: TaskSplit = TaskSplit.VAL,
    max_batches: Optional[int] = None,
) -> BenchmarkResult:
    """
    Evaluate a model on a task using the task's built-in evaluation.

    This simpler path does not require a custom Evaluator subclass.
    """
    model.eval()
    metrics = task.evaluate(model, split=split, max_batches=max_batches)
    params_count = sum(p.numel() for p in model.parameters())

    metric_dict = metrics.to_dict()
    if metric_suite:
        # Accumulate suite metrics across batches
        all_suite_metrics: Dict[str, List[float]] = {}
        loader = task.get_dataloader(split)
        if loader:
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(loader):
                    if max_batches and i >= max_batches:
                        break
                    inputs = inputs.to(task.device)
                    targets = targets.to(task.device)
                    outputs = model(inputs)
                    batch_metrics = metric_suite.evaluate(outputs, targets)
                    for k, v in batch_metrics.items():
                        all_suite_metrics.setdefault(k, []).append(v)
            for k, v in all_suite_metrics.items():
                import numpy as np

                metric_dict[k] = float(np.mean(v))

    return BenchmarkResult(
        model_name=model.__class__.__name__,
        task_name=task.name,
        metrics=metric_dict,
        params_count=params_count,
    )


def cross_validate(
    model_factory: Callable[[], nn.Module],
    task: DomainTask,
    n_folds: int = 5,
    epochs: int = 5,
    metric_suite: Optional[MetricSuite] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run k-fold cross-validation on a task.

    Args:
        model_factory: Callable that returns a fresh model instance.
        task: DomainTask (must support k-fold via DataLoader).
        n_folds: Number of folds.
        epochs: Training epochs per fold.

    Returns:
        Dict of fold -> metrics.
    """
    from bioplausible.core.trainer import CoreTrainer
    from bioplausible.core.trainer import TrainerConfig

    all_fold_metrics: Dict[str, Dict[str, float]] = {}

    for fold in range(n_folds):
        logger.info(f"Cross-validation fold {fold + 1}/{n_folds}")
        model = model_factory()
        config = TrainerConfig(
            model=model.__class__.__name__,
            epochs=epochs,
            task=task.name,
            track_energy=False,
        )
        trainer = CoreTrainer(config)
        trainer.model = model
        trainer.device = task.device
        trainer._setup_data()

        if hasattr(task, "set_fold"):
            task.set_fold(fold, n_folds)

        trainer.fit()
        result = evaluate_model_on_task(model, task, metric_suite=metric_suite)
        all_fold_metrics[f"fold_{fold}"] = result.metrics

    return all_fold_metrics
