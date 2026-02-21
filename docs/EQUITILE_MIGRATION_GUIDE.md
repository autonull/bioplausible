# EquiTile Migration Guide

This guide helps you migrate from legacy EquiTile modules to the new refactored package structure.

## Overview

The EquiTile package has been reorganized into a cleaner, more modular structure:

| Legacy Module | New Module | Notes |
|--------------|------------|-------|
| `equitile.py` | `equitile/core.py` | Core functionality unchanged |
| `equitile_enhanced.py` | `equitile/enhanced.py` | API compatible |
| `equitile_dynamics.py` | `equitile/dynamics.py` | API compatible |
| `equitile_async.py` | `equitile/async_execution.py` | Improved type hints |
| `equitile_multigpu.py` | `equitile/multigpu.py` | Better NCCL support |
| `equitile_distributed.py` | `equitile/distributed.py` | Enhanced mixed precision |
| `equitile_profiler.py` | `equitile/profiler.py` | New memory profiling |

## Quick Migration

### Basic Import Changes

**Before:**
```python
from bioplausible.models.equitile import EquiTile
from bioplausible.models.equitile_enhanced import EnhancedEquiTile
from bioplausible.models.equitile_async import AsyncEquiTile
```

**After:**
```python
from bioplausible.models.equitile import (
    EquiTile,
    EnhancedEquiTile,
    AsyncEquiTile,
)
```

All functionality is now accessible from `bioplausible.models.equitile`.

## Module-Specific Migration

### Async Execution (`equitile_async.py` → `async_execution.py`)

**Before:**
```python
from bioplausible.models.equitile_async import AsyncEquiTile, AsyncConfig

model = EquiTile(...)
async_model = AsyncEquiTile(model, n_workers=4)

with async_model.async_context():
    stats = async_model.train_step(X, y)
```

**After:**
```python
from bioplausible.models.equitile import AsyncEquiTile, AsyncConfig

model = EquiTile(...)
async_model = AsyncEquiTile(model, config=AsyncConfig(n_workers=4))

with async_model.async_context():
    stats = async_model.train_step(X, y)
```

**Changes:**
- `AsyncConfig` is now required (not optional parameters)
- Better type hints throughout
- Improved error messages

### Multi-GPU (`equitile_multigpu.py` → `multigpu.py`)

**Before:**
```python
from bioplausible.models.equitile_multigpu import MultiGPUEquiTile

multi_gpu = MultiGPUEquiTile(model, device_ids=[0, 1, 2, 3])
stats = multi_gpu.train_step(X, y)
```

**After:**
```python
from bioplausible.models.equitile import MultiGPUEquiTile, MultiGPUConfig

multi_gpu = MultiGPUEquiTile(
    model,
    config=MultiGPUConfig(device_ids=[0, 1, 2, 3])
)
stats = multi_gpu.train_step(X, y)
```

**Changes:**
- Configuration via `MultiGPUConfig` dataclass
- Better validation of configuration
- Improved NCCL error handling

### Distributed (`equitile_distributed.py` → `distributed.py`)

**Before:**
```python
from bioplausible.models.equitile_distributed import DistributedEquiTile

dist_model = DistributedEquiTile(
    model,
    device_ids=[0, 1],
    mixed_precision=True,
)
```

**After:**
```python
from bioplausible.models.equitile import DistributedEquiTile, DistributedConfig

dist_model = DistributedEquiTile(
    model,
    config=DistributedConfig(
        device_ids=[0, 1],
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",  # New option
    )
)
```

**Changes:**
- Configuration via `DistributedConfig` dataclass
- Support for `bfloat16` mixed precision
- Better gradient accumulation support

### Profiler (`equitile_profiler.py` → `profiler.py`)

**Before:**
```python
from bioplausible.models.equitile_profiler import EquiTileProfiler

profiler = EquiTileProfiler(model)
with profiler.profile():
    model.train_step(X, y)
profiler.print_report()
```

**After:**
```python
from bioplausible.models.equitile import EquiTileProfiler, MemoryProfiler, BenchmarkRunner

# Basic profiling (same as before)
profiler = EquiTileProfiler(model)
with profiler.profile(batch_size=32):
    model.train_step(X, y)
profiler.print_report()

# New: Memory profiling
memory_profiler = MemoryProfiler(model)
snapshot = memory_profiler.snapshot()
print(f"Memory: {snapshot['total_memory_mb']:.2f} MB")

# New: Benchmarking
from bioplausible.models.equitile import run_benchmark
results = run_benchmark(model, input_dim=784, output_dim=10)
```

**Changes:**
- Added `batch_size` parameter to `profile()`
- New `MemoryProfiler` class
- New `BenchmarkRunner` class
- Better report formatting

## New Features

### Builder Pattern

The new builder pattern provides a fluent API for model construction:

```python
from bioplausible.models.equitile.builder import EquiTileBuilder

# Production model
model = (EquiTileBuilder.production(input_dim=784, output_dim=10)
    .with_learning_rate(0.01)
    .with_dropout(0.1)
    .build())

# Research model with EP
from bioplausible.models.equitile.builder import EnhancedEquiTileBuilder

model = (EnhancedEquiTileBuilder.research(input_dim=784, output_dim=10)
    .enable_layer_norm()
    .enable_curriculum(n_stages=5)
    .build())

# Fast prototyping
model = EquiTileBuilder.fast(input_dim=32, output_dim=10).build()
```

### Context Managers

New context managers for common patterns:

```python
from bioplausible.models.equitile.builder import TrainingContext, InferenceContext

# Training context with automatic logging
with TrainingContext(model, log_interval=100) as ctx:
    for epoch in range(100):
        for X, y in dataloader:
            stats = ctx.train_step(X, y)
        if ctx.should_checkpoint(epoch):
            ctx.save_checkpoint(epoch)

# Inference context
with InferenceContext(model) as ctx:
    predictions = ctx.predict(X)
    probabilities = ctx.predict_proba(X)
    classes = ctx.predict_class(X)
```

### Research Utilities

New research utilities for experiments:

```python
from bioplausible.models.equitile.research import (
    ExperimentTracker,
    MetricCollector,
    VisualizationHelper,
    AblationStudy,
)

# Experiment tracking
tracker = ExperimentTracker("my_experiment")
tracker.log_params({"lr": 0.01, "batch_size": 32})
tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
tracker.save()

# Metric collection
collector = MetricCollector(window_size=100)
collector.add("loss", 0.5)
collector.add("accuracy", 0.9)
print(f"Mean loss: {collector.get_mean('loss')}")
print(f"Trend: {collector.get_trend('loss')}")

# Visualization
viz = VisualizationHelper(model)
viz.plot_activities()
viz.plot_errors()

# Ablation studies
study = AblationStudy(
    name="lr_ablation",
    baseline_params={"lr": 0.01},
    variants=[{"lr": 0.001}, {"lr": 0.1}],
)
study.run_all(train_fn)
```

## Configuration Changes

### Old Style (Deprecated)

```python
model = EquiTile(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
    input_dim=784,
    output_dim=10,
    # Many parameters...
)
```

### New Style (Recommended)

```python
from bioplausible.models.equitile import create_production_config, EquiTile

config = create_production_config(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
)
model = EquiTile(
    neurons_per_tile=config.neurons_per_tile,
    num_layers=config.num_layers,
    tiles_per_layer=config.tiles_per_layer,
    input_dim=784,
    output_dim=10,
)
```

Or use the builder:

```python
from bioplausible.models.equitile.builder import EquiTileBuilder

model = EquiTileBuilder.production(input_dim=784, output_dim=10).build()
```

## Backward Compatibility

The legacy modules are still available but deprecated:

```python
# Still works, but will show deprecation warning
from bioplausible.models.equitile_async import AsyncEquiTile
```

**Recommendation:** Update imports to use the new package structure.

## Testing

After migration, run the test suite:

```bash
# Run all EquiTile tests
python -m pytest tests/test_equitile_refactored.py -v

# Run specific test categories
python -m pytest tests/test_equitile_refactored.py::TestAsyncExecution -v
python -m pytest tests/test_equitile_refactored.py::TestMultiGPU -v
python -m pytest tests/test_equitile_refactored.py::TestProfiler -v
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're using the new package structure:

```python
# Wrong (legacy)
from bioplausible.models.equitile_async import AsyncEquiTile

# Correct (new)
from bioplausible.models.equitile import AsyncEquiTile
```

### Configuration Errors

If you get configuration errors, use the config classes:

```python
# Wrong
async_model = AsyncEquiTile(model, n_workers=4)

# Correct
from bioplausible.models.equitile import AsyncConfig
async_model = AsyncEquiTile(model, config=AsyncConfig(n_workers=4))
```

### Missing Features

If a feature seems missing, check the new modules:

- Memory profiling → `MemoryProfiler` in `profiler.py`
- Benchmarking → `BenchmarkRunner` in `profiler.py`
- Experiment tracking → `ExperimentTracker` in `research.py`
- Visualization → `VisualizationHelper` in `research.py`

## Summary

| Task | Legacy | New |
|------|--------|-----|
| Import | `from equitile_async import` | `from equitile import` |
| Config | Keyword args | Config classes |
| Build | Constructor | Builder pattern |
| Train | Manual loop | `TrainingContext` |
| Profile | `EquiTileProfiler` | + `MemoryProfiler`, `BenchmarkRunner` |
| Research | Manual | `ExperimentTracker`, `AblationStudy` |

For questions or issues, please refer to the API documentation or open an issue.
