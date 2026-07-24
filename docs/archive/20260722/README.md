# EquiTile Package

**Scalable Local-Learning Architecture for Research and Production**

---

## Overview

EquiTile is a tile-based local learning framework that enables:
- **Parallel execution**: Tiles process independently
- **Memory efficiency**: O(1) per tile, no backprop tape
- **Hardware mapping**: GPU, TPU, neuromorphic, edge accelerators
- **Research flexibility**: PC mode (production) and EP mode (research)

---

## Package Structure

```
equitile/
├── __init__.py      # Public API
├── config.py        # Configuration classes
├── core.py          # Core EquiTile implementation
├── distributed.py   # Multi-GPU training (NCCL)
├── enhanced.py      # Enhanced EP (LayerNorm, Curriculum)
├── dynamics.py      # Tile growth/pruning
├── async.py         # Async execution
└── profiler.py      # Profiling and monitoring
```

---

## For Users (Production)

```python
from bioplausible.models.equitile import EquiTile, create_production_config

# Quick setup
model = EquiTile(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=4,
    input_dim=784,
    output_dim=10,
)

# Train
for X, y in dataloader:
    stats = model.train_step(X, y)
```

See `docs/QUICKSTART.md` for more.

---

## For Researchers

### Study Local Learning

```python
from bioplausible.models.equitile import EquiTile

# Compare PC vs EP modes
model_pc = EquiTile(mode='pc', ...)  # Task-driven local Hebbian
model_ep = EquiTile(mode='ep', ...)  # Contrastive EP
```

### Experiment with Architecture

```python
from bioplausible.models.equitile import DynamicEquiTile

# Let architecture adapt during training
dynamic = DynamicEquiTile(model, config=...)
```

### Profile and Analyze

```python
from bioplausible.models.equitile import EquiTileProfiler

profiler = EquiTileProfiler(model)
with profiler.profile():
    model.train_step(X, y)
profiler.print_report()
```

---

## Key Concepts

### Tiles

The network is partitioned into **tiles** - independent compute units:
- Each tile maintains local state (activity, prediction, error)
- Tiles communicate only with immediate neighbors
- Tiles can be processed asynchronously

### Local Learning

Weight updates use only local information:
```
ΔW_ij = η · φ(s_i)ᵀ ⊗ δ_j
```
- `s_i`: Pre-synaptic activity (local)
- `δ_j`: Post-synaptic error (from forward neighbor)
- No global gradient computation

### Two Modes

| Mode | Learning Rule | Performance | Use Case |
|------|---------------|-------------|----------|
| **PC** | Task-driven local Hebbian | 97%+ | Production |
| **EP** | Contrastive (free-nudged) | ~23% | Research |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    EquiTile Model                        │
├─────────────────────────────────────────────────────────┤
│  Input Layer (clamped to data)                          │
│  ┌─────┐ ┌─────┐                                        │
│  │Tile0│ │Tile1│  ...                                  │
│  └─────┘ └─────┘                                        │
│       ↓       ↓                                         │
│  Hidden Layer 1                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
│  │Tile0│ │Tile1│ │Tile2│ │Tile3│  ...                 │
│  └─────┘ └─────┘ └─────┘ └─────┘                       │
│       ↓       ↓       ↓       ↓                         │
│  Hidden Layer 2                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
│  │Tile0│ │Tile1│ │Tile2│ │Tile3│  ...                 │
│  └─────┘ └─────┘ └─────┘ └─────┘                       │
│       ↓       ↓       ↓       ↓                         │
│  Output Layer (task loss)                               │
│  ┌─────┐ ┌─────┐                                        │
│  │Tile0│ │Tile1│  ...                                  │
│  └─────┘ └─────┘                                        │
└─────────────────────────────────────────────────────────┘

Each tile:
┌─────────────────────────────┐
│ Activity (s)                │
│ Prediction (ŝ)              │
│ Error (ε = s - ŝ)           │
│ Importance (learned)        │
└─────────────────────────────┘
```

---

## Research Directions

### 1. Improve EP Convergence

Current EP mode achieves ~23% accuracy vs 97% for PC mode.

**Open questions:**
- Can LayerNorm + curriculum close the gap?
- What initialization schemes work best?
- How many inference steps are needed?

**Start here:** `enhanced.py`, `test_enhanced_ep_layernorm()`

### 2. Tile Dynamics

Automatic architecture adaptation during training.

**Open questions:**
- What growth/pruning thresholds work best?
- Can tiles merge/split dynamically?
- How does dynamics affect convergence?

**Start here:** `dynamics.py`, `test_tile_growth()`

### 3. Multi-GPU Scaling

True async execution across GPUs.

**Open questions:**
- What's the scaling efficiency at 100+ GPUs?
- How to minimize communication overhead?
- Can tiles be distributed heterogeneously?

**Start here:** `distributed.py`, `benchmark_multigpu_scaling()`

### 4. Hardware Mapping

Deploy on specialized hardware.

**Open questions:**
- How to map tiles to neuromorphic cores?
- Can we compile to Edge TPU?
- What's the energy efficiency vs backprop?

**Start here:** Research archive `research/equilibrium_propagation/`

---

## Testing

```bash
# Run all tests
python tests/test_equitile_advanced.py

# Run specific test
python -m pytest tests/test_equitile_advanced.py::test_tile_growth -v
```

---

## Benchmarking

```bash
# Run comprehensive benchmarks
python benchmarks/benchmark_equitile_comprehensive.py

# Results saved to benchmark_results.json
```

---

## Contributing

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `python tests/test_equitile_advanced.py`
5. **Submit PR**

---

## References

- Scellier & Bengio (2017). Equilibrium Propagation.
- Whittington & Bogacz (2017). Predictive Coding as Approximate BP.
- Laborieux et al. (2021). Scaling EP to Deep ConvNets.

---

## License

Same as bioplausible package.
