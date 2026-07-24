# MEP: Muon Equilibrium Propagation

**Version:** 0.3.0
**Status:** Production-ready (core), Experimental (unified optimizer)
**License:** MIT

---

## Overview

MEP is a biologically plausible deep learning framework implementing **Equilibrium Propagation** with **Muon orthogonalization** and **spectral constraints**.

**Key Features:**
- Local learning rules (biologically plausible)
- Stable training at depth (Muon orthogonalization)
- Weight norm control (spectral constraints)
- Compatible with standard PyTorch models

---

## Quick Start

```python
from mep import smep

# Create your model
model = MyNeuralNetwork()

# Create MEP optimizer
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode="ep",  # Use EP (or 'backprop' for comparison)
)

# Training loop
for x, y in train_loader:
    optimizer.step(x=x, target=y)
```

---

## Performance

| Benchmark | MEP | Backprop | Notes |
|-----------|-----|----------|-------|
| MNIST (3 epochs) | 91-94% | 90-93% | MEP matches BP |
| MNIST (10 epochs) | 95-96% | 95-96% | Parity |
| XOR (100 steps) | ≥95% | 100% | Both solve XOR |
| Speed | 2-3x slower | 1.0x | EP settling overhead |

---

## Installation

```bash
# From this repository
pip install -e .

# Requirements
torch >= 2.0
torchvision >= 0.15
```

---

## Usage

### Basic Usage

```python
from mep import smep

optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,  # Learning rate
    mode="ep",  # 'ep' or 'backprop'
    settle_steps=30,  # EP settling iterations
    settle_lr=0.15,  # Settling learning rate
    beta=0.5,  # Nudging strength
    loss_type="mse",  # 'mse' or 'cross_entropy'
)

for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.step(x=x, target=y)
```

### Advanced Usage

```python
from mep import smep

# High-accuracy configuration
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    settle_steps=30,  # More settling = better accuracy
    settle_lr=0.15,
    beta=0.5,
    loss_type="mse",
    gamma=0.95,  # Spectral norm bound
    ns_steps=5,  # Muon orthogonalization steps
)

# Fast prototyping configuration
from mep import smep_fast

optimizer = smep_fast(
    model.parameters(),
    model=model,
    settle_steps=10,  # Fewer settling = faster
)
```

### Backprop Comparison

```python
from mep import muon_backprop

# Use Muon with standard backprop
optimizer = muon_backprop(
    model.parameters(),
    lr=0.02,
)

for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

---

## Architecture

```
mep/
├── optimizers/
│   ├── composite.py       # Strategy pattern optimizer
│   ├── strategies/
│   │   ├── gradient.py    # EPGradient, BackpropGradient
│   │   ├── update.py      # MuonUpdate, DionUpdate
│   │   ├── constraint.py  # SpectralConstraint
│   │   └── feedback.py    # ErrorFeedback, NoFeedback
│   ├── settling.py        # Settling dynamics
│   ├── energy.py          # EP energy computation
│   └── inspector.py       # Model structure extraction
├── presets/
│   └── __init__.py        # smep, smep_fast, etc.
└── cuda/
    └── kernels.py         # Optional CUDA kernels
```

---

## Testing

```bash
# Smoke test (< 20 seconds)
python tests/regression/test_ep_smoke.py

# Full regression (< 3 minutes)
python tests/regression/test_ep_baseline.py

# Performance benchmarks
python -m pytest tests/regression/test_performance_baseline.py -xvs
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `INTEGRATION_GUIDE.md` | Integration into Bioplausible |
| `docs/benchmarks/PERFORMANCE_BASELINES.md` | Performance baselines |
| `docs/benchmarks/performance_report_corrected.md` | Performance analysis |
| `docs/research/phase2_summary.md` | Phase 2 technical summary |
| `docs/development/workflow.md` | Development workflow |

---

## Known Limitations

1. **Speed:** EP is 2-3x slower than backprop (fundamental settling cost)
2. **Memory:** EP uses more memory than backprop+checkpointing
3. **Dropout:** Incompatible with EP settling (skip dropout during settling)

---

## Research Directions

1. **Adaptive settling** - Early stopping when converged
2. **Custom CUDA kernels** - Fused settling operations
3. **Continual learning** - EP + EWC integration
4. **Neuromorphic deployment** - Event-based computation

---

## Citation

```bibtex
@software{mep2026,
  title = {MEP: Muon Equilibrium Propagation},
  version = {0.3.0},
  year = {2026},
  url = {https://github.com/automenta/mep}
}
```

---

## License

MIT License - see LICENSE file for details.

---

*Last updated: 2026-02-19*
