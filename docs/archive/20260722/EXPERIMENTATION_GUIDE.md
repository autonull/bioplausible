# Experimentation Guide

**Date:** 2026-02-19  
**Status:** Ready for research use

---

## Overview

The Bioplausible framework provides comprehensive utilities for experimentation and research on biologically plausible learning algorithms. This guide covers:

- Quick experimentation with presets
- Custom experiment design
- Hyperparameter search
- Model/optimizer comparison
- Validation and benchmarking

---

## Quick Start

### 1. Run a Preset Experiment

```python
from bioplausible import get_preset, run_preset, list_presets
from bioplausible.datasets import get_vision_dataset

# See available presets
print(list_presets())  # All presets
print(list_presets("performance"))  # Performance category

# Load data
train_loader, val_loader, _ = get_vision_dataset("mnist", batch_size=128)

# Run preset
result = run_preset("performance_vision_default", train_loader, val_loader)
print(result.summary())
```

### 2. Compare Optimizers

```python
from bioplausible.experiments import quick_comparison

results = quick_comparison(
    model_name="looped_mlp",
    optimizer_names=["smep", "smep_fast", "muon_backprop"],
    epochs=3,
)

for r in results:
    print(f"{r.optimizer_name}: {r.val_accuracy:.2f}%")
```

### 3. Benchmark a Model

```python
from bioplausible.experiments import benchmark_model

result = benchmark_model(
    model_name="conv_eqprop",
    optimizer_name="smep",
    epochs=10,
)

print(f"Test Accuracy: {result.test_accuracy:.2f}%")
print(f"Training Speed: {result.steps_per_second:.1f} steps/s")
```

---

## Research Presets

Presets are pre-configured model/optimizer combinations organized by research goal.

### Categories

| Category | Description | Use Case |
|----------|-------------|----------|
| `performance` | Best accuracy | Final experiments, SOTA comparison |
| `speed` | Fast training | Prototyping, hyperparameter search |
| `efficiency` | Memory/compute efficient | Deep nets, edge deployment |
| `bioplausible` | Most biologically plausible | Biological modeling |
| `robustness` | Noise/distribution robust | Adversarial training |
| `exploratory` | Experimental configurations | Novel research directions |

### Available Presets

#### Performance

| Preset | Model | Optimizer | Expected Accuracy |
|--------|-------|-----------|-------------------|
| `performance_vision_default` | looped_mlp | smep | 95-97% MNIST |
| `performance_vision_cnn` | modern_conv_eqprop | smep | 70-80% CIFAR-10 |
| `performance_lm` | transformer_eqprop | smep | 1.5-2.0 BPC |

#### Speed

| Preset | Model | Optimizer | Speed |
|--------|-------|-----------|-------|
| `speed_vision_fast` | looped_mlp | smep_fast | 4-6x slower than BP |
| `speed_backprop_baseline` | looped_mlp | muon_backprop | 1.2x slower than BP |

#### Bioplausible

| Preset | Model | Optimizer | Description |
|--------|-------|-----------|-------------|
| `bioplausible_local` | looped_mlp | local_ep | Layer-local learning |
| `bioplausible_hebbian` | hebbian_chain | muon_backprop | Pure Hebbian |
| `bioplausible_feedback` | feedback_alignment | muon_backprop | Random feedback |

### Using Presets

```python
from bioplausible.experiments import get_preset, ALL_PRESETS

# Get specific preset
preset = get_preset("performance_vision_default")
print(f"Model: {preset.model_name}")
print(f"Optimizer: {preset.optimizer_name}")
print(f"Params: {preset.model_params}")

# Filter by category
from bioplausible.experiments import get_preset_by_category

performance_presets = get_preset_by_category("performance")
```

---

## Experiment Runner

The `ExperimentRunner` class provides controlled experiment execution.

### Basic Usage

```python
from bioplausible.experiments import ExperimentRunner

runner = ExperimentRunner(device="cuda")

result = runner.run(
    model_name="looped_mlp",
    optimizer_name="smep",
    train_loader=train_loader,
    val_loader=val_loader,
    model_params={"input_dim": 784, "hidden_dim": 512},
    optimizer_params={"lr": 0.01, "settle_steps": 30},
    epochs=10,
    batches_per_epoch=100,
    verbose=True,
)
```

### Result Object

```python
print(f"Train Accuracy: {result.train_accuracy:.2f}%")
print(f"Val Accuracy: {result.val_accuracy:.2f}%")
print(f"Test Accuracy: {result.test_accuracy:.2f}%")
print(f"Training Time: {result.training_time:.1f}s")
print(f"Steps/Second: {result.steps_per_second:.1f}")
print(f"Parameters: {result.num_parameters:,}")
```

### Comparing Optimizers

```python
results = runner.compare_optimizers(
    model_name="looped_mlp",
    optimizer_names=["smep", "smep_fast", "sdmep", "local_ep"],
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
)

# Results sorted by validation accuracy
for i, r in enumerate(results):
    print(f"{i + 1}. {r.optimizer_name}: {r.val_accuracy:.2f}%")
```

### Comparing Models

```python
results = runner.compare_models(
    model_names=["looped_mlp", "conv_eqprop", "memory_efficient_mlp"],
    optimizer_name="smep",
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
)

for i, r in enumerate(results):
    print(f"{i + 1}. {r.model_name}: {r.val_accuracy:.2f}%")
```

---

## Hyperparameter Search

### Grid Search

```python
from bioplausible.experiments import HyperparameterSearch

search = HyperparameterSearch(device="cuda")

best_params, best_result = search.grid_search(
    model_name="looped_mlp",
    optimizer_name="smep",
    param_grid={
        "lr": [0.001, 0.01, 0.1],
        "settle_steps": [10, 30, 50],
        "beta": [0.3, 0.5, 0.7],
    },
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
    verbose=True,
)

print(f"Best params: {best_params}")
print(f"Best accuracy: {best_result.val_accuracy:.2f}%")
```

### Custom Search Strategy

```python
from bioplausible.zoo import ModelZoo, OptimizerZoo
from bioplausible.experiments import ExperimentRunner
import numpy as np

runner = ExperimentRunner()

# Random search
best_acc = 0
best_config = None

for _ in range(20):
    lr = 10 ** np.random.uniform(-4, -1)
    beta = np.random.uniform(0.2, 0.8)
    settle_steps = np.random.choice([10, 20, 30, 50])

    result = runner.run(
        model_name="looped_mlp",
        optimizer_name="smep",
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_params={
            "lr": lr,
            "beta": beta,
            "settle_steps": settle_steps,
        },
        epochs=3,
        verbose=False,
    )

    if result.val_accuracy > best_acc:
        best_acc = result.val_accuracy
        best_config = {"lr": lr, "beta": beta, "settle_steps": settle_steps}

print(f"Best random config: {best_config}")
print(f"Best accuracy: {best_acc:.2f}%")
```

---

## Validation Workflows

### Smoke Test (20 seconds)

```python
from bioplausible.experiments import ExperimentRunner

runner = ExperimentRunner()

# Quick sanity check
result = runner.run(
    model_name="looped_mlp",
    optimizer_name="smep",
    train_loader=train_loader,
    epochs=1,
    batches_per_epoch=10,
    verbose=False,
)

assert result.train_accuracy > 15, "Model not learning!"
print("✓ Smoke test passed")
```

### Extended Validation (3 minutes)

```python
result = runner.run(
    model_name="looped_mlp",
    optimizer_name="smep",
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
    verbose=True,
)

# Validation criteria
assert result.val_accuracy > 85, f"Low accuracy: {result.val_accuracy}"
assert not np.isnan(result.train_loss), "NaN in training"
assert result.steps_per_second > 0, "Training stalled"

print("✓ Extended validation passed")
```

### Full Benchmark (30 minutes)

```python
from bioplausible.experiments import benchmark_model

result = benchmark_model(
    model_name="looped_mlp",
    optimizer_name="smep",
    epochs=10,
)

# Comprehensive validation
print(f"Final Test Accuracy: {result.test_accuracy:.2f}%")
print(f"Training Speed: {result.steps_per_second:.1f} steps/s")
print(f"Memory Peak: {result.memory_peak_mb:.1f} MB")
```

---

## Custom Experiment Design

### Creating Custom Presets

```python
from bioplausible.experiments.presets import ResearchPreset

# Define your preset
my_preset = ResearchPreset(
    name="my_custom_experiment",
    category="exploratory",
    model_name="looped_mlp",
    model_params={"input_dim": 784, "hidden_dim": 1024},
    optimizer_name="smep",
    optimizer_params={"lr": 0.005, "settle_steps": 40},
    description="My custom high-capacity model",
    use_case="Testing larger hidden dimensions",
    tags=["custom", "high-capacity"],
)

# Use it
from bioplausible.experiments import ExperimentRunner

runner = ExperimentRunner()
result = runner.run(
    model_name=my_preset.model_name,
    optimizer_name=my_preset.optimizer_name,
    train_loader=train_loader,
    val_loader=val_loader,
    model_params=my_preset.model_params,
    optimizer_params=my_preset.optimizer_params,
    epochs=10,
)
```

### Ablation Studies

```python
from bioplausible.experiments import ExperimentRunner

runner = ExperimentRunner()

# Study effect of settle_steps
results = []
for steps in [5, 10, 20, 40, 60]:
    result = runner.run(
        model_name="looped_mlp",
        optimizer_name="smep",
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_params={"settle_steps": steps},
        epochs=5,
        verbose=False,
    )
    results.append((steps, result.val_accuracy, result.training_time))

print("Settle Steps | Val Accuracy | Time")
print("-" * 40)
for steps, acc, time in results:
    print(f"{steps:14} | {acc:12.2f}% | {time:.1f}s")
```

### Scaling Studies

```python
# Study effect of model size
hidden_sizes = [64, 128, 256, 512, 1024]
results = []

for hidden in hidden_sizes:
    result = runner.run(
        model_name="looped_mlp",
        optimizer_name="smep",
        train_loader=train_loader,
        val_loader=val_loader,
        model_params={"hidden_dim": hidden},
        epochs=5,
        verbose=False,
    )
    results.append((hidden, result.num_parameters, result.val_accuracy))

print("Hidden | Parameters | Val Accuracy")
print("-" * 40)
for hidden, params, acc in results:
    print(f"{hidden:6} | {params:10,} | {acc:12.2f}%")
```

---

## Best Practices

### 1. Start with Presets

Begin with validated presets before customizing:
```python
# Good starting point
result = run_preset("speed_vision_fast", train_loader, val_loader, epochs=3)
```

### 2. Use Appropriate Epochs

| Purpose | Epochs |
|---------|--------|
| Smoke test | 1 |
| Hyperparameter search | 3-5 |
| Validation | 5-10 |
| Final benchmark | 10-50 |

### 3. Monitor Multiple Metrics

```python
result = runner.run(...)

# Check multiple aspects
assert result.val_accuracy > threshold, "Low accuracy"
assert result.steps_per_second > min_speed, "Too slow"
assert result.num_parameters < max_params, "Too large"
```

### 4. Use Validation Set

Always use a separate validation set for hyperparameter tuning:
```python
result = runner.run(
    ...,
    train_loader=train_loader,
    val_loader=val_loader,  # For tuning
    test_loader=test_loader,  # For final evaluation
)
```

### 5. Document Configurations

```python
from dataclasses import asdict

config = {
    "model": result.model_name,
    "optimizer": result.optimizer_name,
    "model_params": result.model_params,
    "optimizer_params": result.optimizer_params,
    "epochs": 10,
}

import json

with open("experiment_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

---

## Troubleshooting

### Model Not Learning

```python
# Check learning rate
result = runner.run(
    ...,
    optimizer_params={"lr": 0.01},  # Try 0.001 or 0.1
    verbose=True,
)

# Check for NaN
if np.isnan(result.train_loss):
    print("NaN detected - reduce learning rate or beta")
```

### Training Too Slow

```python
# Use faster optimizer
result = runner.run(
    ...,
    optimizer_name="smep_fast",  # 4-6x faster
    optimizer_params={"settle_steps": 10},  # Fewer steps
)
```

### Out of Memory

```python
# Use memory-efficient model
result = runner.run(
    model_name='memory_efficient_mlp',  # Gradient checkpointing
    ...,
    optimizer_params={'settle_steps': 10},
)

# Reduce batch size
train_loader = DataLoader(dataset, batch_size=32)  # Instead of 128
```

---

## API Reference

### ExperimentRunner

```python
runner = ExperimentRunner(device='auto')

result = runner.run(
    model_name: str,
    optimizer_name: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model_params: Dict[str, Any],
    optimizer_params: Dict[str, Any],
    epochs: int,
    batches_per_epoch: int,
    verbose: bool,
) -> ExperimentResult
```

### HyperparameterSearch

```python
search = HyperparameterSearch(device='auto')

best_params, best_result = search.grid_search(
    model_name: str,
    optimizer_name: str,
    param_grid: Dict[str, List[Any]],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    verbose: bool,
) -> Tuple[Dict[str, Any], ExperimentResult]
```

### ExperimentResult

```python
@dataclass
class ExperimentResult:
    model_name: str
    optimizer_name: str
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_loss: float
    val_loss: float
    training_time: float
    steps_per_second: float
    num_parameters: int
    memory_peak_mb: float
```

---

## References

- [MEP Integration Summary](MEP_INTEGRATION_SUMMARY.md)
- [MEP Integration Guide](MEP_INTEGRATION.md)
- [Scientist Guide](../SCIENTIST_GUIDE.md)

---

*Created: 2026-02-19*  
*Status: Ready for research use*
