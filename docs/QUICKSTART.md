# EquiTile Quickstart Guide

Welcome to EquiTile! This guide will help you get started in 5 minutes.

---

## Installation

```bash
# EquiTile is part of the bioplausible package
pip install bioplausible

# Or use from source
cd /path/to/biopl
export PYTHONPATH=/path/to/biopl:$PYTHONPATH
```

---

## 1. Basic Usage (30 seconds)

```python
from bioplausible.models.equitile import EquiTile

# Create model
model = EquiTile(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
    input_dim=784,  # e.g., MNIST
    output_dim=10,
)

# Train
for X, y in dataloader:
    stats = model.train_step(X, y)
    print(f"Loss: {stats['loss']:.4f}, Acc: {stats['accuracy']:.4f}")
```

---

## 2. Production Configuration (1 minute)

```python
from bioplausible.models.equitile import EquiTile, create_production_config

# Use production config factory
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

# Configure learning rate scheduler
model.configure_lr_scheduler(
    scheduler_type="cosine",
    total_steps=10000,
    warmup_steps=1000,
)

# Training loop with checkpointing
for epoch in range(100):
    for X, y in dataloader:
        stats = model.train_step(X, y)
        model.step_lr_scheduler()
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        model.save_checkpoint(f"checkpoint_epoch{epoch}.pt")
```

---

## 3. Multi-GPU Training (2 minutes)

```python
from bioplausible.models.equitile import EquiTile, DistributedEquiTile, MultiGPUConfig

# Create base model
model = EquiTile(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
    input_dim=784,
    output_dim=10,
)

# Wrap for multi-GPU
multi_gpu = DistributedEquiTile(
    model,
    config=MultiGPUConfig(
        device_ids=[0, 1, 2, 3],  # Use 4 GPUs
        async_execution=True,
    )
)

# Train (same API!)
for X, y in dataloader:
    stats = multi_gpu.train_step(X, y)
```

---

## 4. Enhanced EP for Research (2 minutes)

```python
from bioplasible.models.equitile import (
    EquiTile,
    EnhancedEquiTile,
    create_enhanced_config,
)

# Create base model in EP mode
model = EquiTile(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
    input_dim=784,
    output_dim=10,
    mode='ep',  # EP mode for research
)

# Wrap with enhanced features
enhanced = EnhancedEquiTile(
    model,
    config=create_enhanced_config(
        use_layer_norm=True,
        use_curriculum=True,
        curriculum_stages=5,
    )
)

# Train with curriculum
for X, y in dataloader:
    stats = enhanced.train_step(X, y)
    enhanced.curriculum.step(stats['loss'])
```

---

## 5. Dynamic Tile Architecture (2 minutes)

```python
from bioplasible.models.equitile import (
    EquiTile,
    DynamicEquiTile,
    create_dynamic_config,
)

# Create model
model = EquiTile(
    neurons_per_tile=32,
    num_layers=3,
    tiles_per_layer=2,
    input_dim=64,
    output_dim=4,
)

# Wrap with dynamics
dynamic = DynamicEquiTile(
    model,
    config=create_dynamic_config(
        growth_enabled=True,
        prune_enabled=True,
        growth_threshold=0.5,
        prune_threshold=0.05,
    )
)

# Train with automatic tile modification
for X, y in dataloader:
    stats = model.train_step(X, y)
    
    # Check for tile modifications
    mods = dynamic.step()
    if mods['grown'] > 0:
        print(f"Grew {mods['grown']} tiles")
    if mods['pruned'] > 0:
        print(f"Pruned {mods['pruned']} tiles")
```

---

## 6. Monitoring and Profiling (1 minute)

```python
from bioplasible.models.equitile import EquiTile, LearningMonitor

model = EquiTile(...)
monitor = LearningMonitor(model, window_size=100)

for X, y in dataloader:
    stats = model.train_step(X, y)
    monitor.record(stats)
    
    # Print status every 10 steps
    if step % 10 == 0:
        monitor.print_status()

# Get summary
summary = monitor.get_summary()
print(f"Loss trend: {summary['loss_trend']}")
print(f"Hot tiles: {summary['hot_tiles']}")
```

---

## Common Patterns

### Classification

```python
model = EquiTile(
    input_dim=784,
    output_dim=10,
    task_type="classification",  # Cross-entropy loss
)
```

### Regression

```python
model = EquiTile(
    input_dim=20,
    output_dim=1,
    task_type="regression",  # MSE loss, R² metric
)
```

### Binary Classification

```python
model = EquiTile(
    input_dim=100,
    output_dim=1,
    task_type="binary",  # Sigmoid + BCE
)
```

---

## Next Steps

1. **Read the full docs**: `docs/EquiTile_COMPLETE.md`
2. **Run examples**: `examples/equitile_production_training.py`
3. **Run benchmarks**: `benchmarks/benchmark_equitile_comprehensive.py`
4. **Check research archive**: `research/equilibrium_propagation/`

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `neurons_per_tile` or `tiles_per_layer`
- Enable mixed precision: `with torch.amp.autocast('cuda')`

### "EP mode not converging"
- Use PC mode for production: `mode='pc'`
- Enable LayerNorm: `use_layer_norm=True`
- Use curriculum learning

### "Multi-GPU not scaling"
- Check NCCL initialization
- Increase batch size
- Reduce sync frequency

---

## Getting Help

- **Documentation**: `docs/`
- **Examples**: `examples/`
- **Tests**: `tests/test_equitile_advanced.py`
- **Status**: `EQUITILE_STATUS.md`
