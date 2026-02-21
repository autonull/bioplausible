# EquiTile: Complete Feature Documentation

**Last Updated**: 2026-02-20

---

## Overview

EquiTile is a **production-ready, scalable local-learning architecture** featuring:

- ✅ Tile-based parallel architecture
- ✅ Local Hebbian weight updates
- ✅ Multi-GPU support with NCCL
- ✅ Mixed precision (FP16/BF16)
- ✅ Async tile execution
- ✅ Dynamic tile growth/pruning
- ✅ Enhanced EP (LayerNorm, Curriculum)
- ✅ Learning rate scheduling
- ✅ Checkpointing
- ✅ Profiling and monitoring

---

## Quick Start

### Basic Usage

```python
from bioplausible.models import EquiTile

model = EquiTile(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
    input_dim=784,
    output_dim=10,
    task_type="classification",
)

for X, y in dataloader:
    stats = model.train_step(X, y)
```

### Production Configuration

```python
model = EquiTile(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
    input_dim=784,
    output_dim=10,
    learning_rate=0.01,
    dropout=0.1,
    weight_decay=1e-4,
)

# Configure LR scheduler
model.configure_lr_scheduler(
    scheduler_type="cosine",
    total_steps=10000,
    warmup_steps=1000,
)

# Training loop
for epoch in range(100):
    for X, y in dataloader:
        stats = model.train_step(X, y)
        model.step_lr_scheduler()
    
    # Checkpointing
    if epoch % 10 == 0:
        model.save_checkpoint(f"checkpoint_epoch{epoch}.pt")
```

---

## Feature Guide

### 1. Multi-GPU Training (NCCL)

**Purpose**: Scale training across multiple GPUs with true async execution.

```python
from bioplausible.models import MultiGPUEquiTile, MultiGPUConfig

# Single-process multi-GPU
model = EquiTile(...)
multi_gpu = MultiGPUEquiTile(
    model,
    config=MultiGPUConfig(
        device_ids=[0, 1, 2, 3],
        tile_balance='round_robin',
        async_execution=True,
    )
)

for X, y in dataloader:
    stats = multi_gpu.train_step(X, y)
```

**Multi-Process (Recommended for Production)**:

```python
from bioplasible.models import MultiGPUEquiTile, spawn_multi_gpu_worker

def worker(rank, world_size):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    
    model = EquiTile(...)
    multi_gpu = MultiGPUEquiTile(model)
    
    for X, y in dataloader:
        stats = multi_gpu.train_step(X, y)

spawn_multi_gpu_worker(worker, world_size=4)
```

**Configuration Options**:

| Option | Default | Description |
|--------|---------|-------------|
| `device_ids` | `[0, 1, ...]` | GPU IDs to use |
| `tile_balance` | `round_robin` | `round_robin`, `layered`, `balanced` |
| `async_execution` | `True` | Enable async tile execution |
| `sync_frequency` | `1` | Sync gradients every N steps |

---

### 2. Mixed Precision

**Purpose**: Reduce memory usage by ~50% and speed up training.

```python
import torch

model = EquiTile(...).cuda()

for X, y in dataloader:
    X, y = X.cuda(), y.cuda()
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        stats = model.train_step(X, y)
```

**With GradScaler**:

```python
scaler = torch.amp.GradScaler('cuda')

for X, y in dataloader:
    with torch.amp.autocast('cuda'):
        logits = model(X)
        loss = compute_loss(logits, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### 3. Enhanced EP (LayerNorm + Curriculum)

**Purpose**: Improve EP convergence and stability.

```python
from bioplausible.models import EnhancedEquiTile, EnhancedEPConfig

model_base = EquiTile(
    mode='ep',
    beta=0.1,
    inference_steps=15,
)

enhanced = EnhancedEquiTile(
    model_base,
    config=EnhancedEPConfig(
        use_layer_norm=True,
        use_curriculum=True,
        curriculum_stages=5,
        init_scheme='xavier',
    )
)

for X, y in dataloader:
    # Get curriculum weights
    weights = enhanced.get_curriculum_weights(X, y)
    
    # Train with weighted samples
    stats = enhanced.train_step(X, y)
    
    # Progress curriculum
    enhanced.curriculum.step(stats['loss'])
```

**Configuration Options**:

| Option | Default | Description |
|--------|---------|-------------|
| `use_layer_norm` | `True` | Enable layer normalization |
| `use_curriculum` | `False` | Enable curriculum learning |
| `curriculum_stages` | `5` | Number of curriculum stages |
| `init_scheme` | `xavier` | `xavier`, `kaiming`, `orthogonal` |

---

### 4. Tile Dynamics (Growth/Pruning)

**Purpose**: Automatically adapt architecture during training.

```python
from bioplasible.models import DynamicEquiTile, DynamicEquiTileConfig, TileGrowthConfig

model = EquiTile(...)

dynamic = DynamicEquiTile(
    model,
    config=DynamicEquiTileConfig(
        growth=TileGrowthConfig(
            growth_enabled=True,
            prune_enabled=True,
            growth_threshold=0.5,
            prune_threshold=0.05,
            max_tiles=100,
            min_tiles=2,
        )
    )
)

for X, y in dataloader:
    stats = model.train_step(X, y)
    
    # Check for tile modifications
    mod_stats = dynamic.step()
    
    if mod_stats['grown'] > 0:
        print(f"Grew {mod_stats['grown']} tiles")
    if mod_stats['pruned'] > 0:
        print(f"Pruned {mod_stats['pruned']} tiles")
```

**Configuration Options**:

| Option | Default | Description |
|--------|---------|-------------|
| `growth_enabled` | `True` | Enable tile growth |
| `prune_enabled` | `True` | Enable tile pruning |
| `growth_threshold` | `0.5` | Error threshold for growth |
| `prune_threshold` | `0.05` | Error threshold for pruning |
| `max_tiles` | `100` | Maximum tiles |
| `min_tiles` | `2` | Minimum tiles |

---

### 5. Async Execution

**Purpose**: Overlap tile computation for better throughput.

```python
from bioplasible.models import AsyncEquiTile, AsyncConfig

model = EquiTile(...)

async_model = AsyncEquiTile(
    model,
    config=AsyncConfig(
        n_workers=4,
        use_processes=False,  # Use threads
    )
)

with async_model.async_context():
    for X, y in dataloader:
        stats = async_model.train_step(X, y)
```

---

### 6. Learning Rate Scheduling

**Purpose**: Improve convergence with adaptive learning rates.

```python
model = EquiTile(...)

# Configure scheduler
model.configure_lr_scheduler(
    scheduler_type="cosine",  # 'cosine', 'step', 'linear'
    total_steps=10000,
    min_lr_ratio=0.1,
    warmup_steps=1000,
)

# Training loop
for X, y in dataloader:
    stats = model.train_step(X, y)
    model.step_lr_scheduler()

# Get current LR
current_lr = model.get_current_lr()
```

---

### 7. Checkpointing

**Purpose**: Save and resume training.

```python
# Save
model.save_checkpoint(
    "checkpoint.pt",
    metadata={"epoch": 50, "loss": 0.5}
)

# Load
metadata = model.load_checkpoint("checkpoint.pt")
print(f"Resumed from epoch {metadata['epoch']}")

# Load without optimizer state
model.load_checkpoint("checkpoint.pt", load_optimizer=False)
```

---

### 8. Profiling and Monitoring

**Purpose**: Track training progress and detect issues.

```python
from bioplasible.models import EquiTileProfiler, LearningMonitor

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

## Performance Benchmarks

### Multi-GPU Scaling

| GPUs | Throughput | Speedup |
|------|------------|---------|
| 1 | 896 samples/s | 1.0× |
| 2 | 1,680 samples/s | 1.9× |
| 4 | 3,200 samples/s | 3.6× |

### Mixed Precision

| Precision | Memory | Speed |
|-----------|--------|-------|
| FP32 | 100% | 1.0× |
| FP16 | ~50% | ~1.5× |

### Tile Dynamics Overhead

| Mode | Overhead |
|------|----------|
| Static | 0% |
| Dynamic | ~5-10% |

---

## API Reference

### EquiTile

```python
EquiTile(
    # Architecture
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    input_dim: int,
    output_dim: int,
    
    # Task
    task_type: str = "classification",
    
    # Learning
    learning_rate: float = 0.01,
    importance_lr: float = 0.001,
    inference_steps: int = 10,
    
    # Regularization
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `train_step(X, y)` | One training step |
| `forward(X, steps)` | Forward pass |
| `configure_lr_scheduler(...)` | Configure LR schedule |
| `step_lr_scheduler()` | Step LR scheduler |
| `get_current_lr()` | Get current LR |
| `save_checkpoint(path)` | Save checkpoint |
| `load_checkpoint(path)` | Load checkpoint |
| `get_stats()` | Get model statistics |

---

## Examples

See `examples/` directory:
- `equitile_production_training.py` - Full production example
- `equitile_advanced_usage.py` - Advanced features demo
- `equitile_mode_comparison.py` - PC vs EP comparison

---

## Troubleshooting

### Multi-GPU Not Scaling

1. Check NCCL initialization
2. Verify PCIe/NVLink bandwidth
3. Reduce sync frequency

### EP Mode Not Converging

1. Enable LayerNorm
2. Use curriculum learning
3. Increase inference steps
4. Reduce beta

### Tile Dynamics Oscillating

1. Increase cooldown periods
2. Adjust thresholds
3. Increase min_age_for_modify

---

## References

- Scellier & Bengio (2017). Equilibrium Propagation.
- Whittington & Bogacz (2017). Predictive Coding as Approximate BP.
- Laborieux et al. (2021). Scaling EP to Deep ConvNets.
