# Contributing a New Domain to Bioplausible

Adding a new domain (vision, LM, RL, graph, timeseries, tabular, scientific) to Bioplausible
takes less than 1 day for experienced contributors.

## Overview

All domains share a common interface via `DomainTask`. Your implementation must:
1. Inherit from `DomainTask`
2. Implement `setup()`, `get_dataloader()`, and `evaluate()` methods
3. Define a `DomainSpec` with domain metadata
4. Register benchmarks in the benchmark registry

## Quick Start Template

Create `bioplausible/domains/my_domain.py`:

```python
from typing import Optional
import torch.nn as nn
from bioplausible.domains.base import (
    DomainSpec,
    DomainTask,
    DomainType,
    TaskSplit,
    Metrics,
)

class MyDomainTask(DomainTask):
    @property
    def domain_type(self) -> DomainType:
        return DomainType.CUSTOM  # Or appropriate enum

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name="my_domain",
            domain_type=DomainType.CUSTOM,
            description="My custom domain",
            default_metrics=["accuracy"],
            supported_tasks=["task_a", "task_b"],
            default_batch_size=32,
        )

    def setup(self) -> None:
        # Load your dataset
        self._train_loader = ...
        self._input_dim = ...
        self._output_dim = ...

    def get_dataloader(self, split: TaskSplit):
        # Return appropriate DataLoader
        ...

    def evaluate(self, model: nn.Module, split: TaskSplit = TaskSplit.VAL, max_batches: int = None) -> Metrics:
        # Evaluate and return standardized metrics
        ...
```

## Register Your Task

Add to `bioplausible/domains/__init__.py`:
```python
from bioplausible.domains.my_domain import MyDomainTask
_DOMAIN_REGISTRY["my_domain"] = MyDomainTask
```

## Add Benchmarks

Add to `bioplausible/evaluation/benchmarks.py`:
```python
def my_task_benchmark(model: nn.Module) -> BenchmarkResult:
    task = MyDomainTask(name="my_task")
    task.setup()
    return evaluate_model_on_task(model, task)

BenchmarkRegistry.register("my_task", my_task_benchmark)
```

## Testing

Create `tests/test_my_domain.py`:
```python
def test_my_domain_task():
    task = MyDomainTask(...)
    task.setup()
    assert task.input_dim > 0

def test_my_domain_benchmark():
    model = nn.Linear(10, 2)
    result = my_task_benchmark(model)
    assert "accuracy" in result.metrics
```

## Configuration

Add to `bioplausible/config/schema.py` if needed.