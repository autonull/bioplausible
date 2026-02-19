# MEP Documentation

Welcome to the MEP (Muon Equilibrium Propagation) documentation.

## Quick Start

**New to MEP?** Start here:
1. [README.md](../README.md) - Installation and quick start
2. [Getting Started](#getting-started) - Basic usage guide
3. [Optimizer Selection Guide](#optimizer-selection) - Choose the right optimizer

---

## 📚 Documentation Index

### Getting Started
| Document | Description |
|----------|-------------|
| [README.md](../README.md) | Installation, quick start, basic usage |
| [examples/quickstart.py](../examples/quickstart.py) | Minimal working example |
| [examples/demo_ep_vs_backprop.py](../examples/demo_ep_vs_backprop.py) | EP vs backprop comparison demo |

### Benchmarks & Performance
| Document | Description |
|----------|-------------|
| [benchmarks/PERFORMANCE_BASELINES.md](benchmarks/PERFORMANCE_BASELINES.md) | **Start here** - Performance thresholds, optimal config |
| [benchmarks/VALIDATION_RESULTS.md](benchmarks/VALIDATION_RESULTS.md) | Full validation study with findings |
| [benchmarks/ci_configuration.md](benchmarks/ci_configuration.md) | CI benchmark setup guide |
| [tests/regression/test_performance_baseline.py](../tests/regression/test_performance_baseline.py) | Automated regression tests |

### Research & Roadmap
| Document | Description |
|----------|-------------|
| [research/ROADMAP_RESEARCH.md](research/ROADMAP_RESEARCH.md) | **Start here** - Research trajectory, partnerships |
| [research/ROADMAP.md](research/ROADMAP.md) | Development roadmap with milestones |
| [methods_paper.md](methods_paper.md) | Preprint-ready methods paper |
| [outreach_plan.md](outreach_plan.md) | Research collaboration strategy |

### API Reference
| Document | Description |
|----------|-------------|
| [API Reference](#api-reference) | Optimizer classes and functions |
| [mep/optimizers/](../mep/optimizers/) | Optimizer implementation |
| [mep/presets/](../mep/presets/) | Preset optimizer configurations |

---

## 🚀 Getting Started

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
import torch.nn as nn
from mep import smep, muon_backprop

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Option 1: EP mode (biologically plausible)
optimizer = smep(model.parameters(), model=model, mode='ep')
optimizer.step(x=x, target=y)  # No .backward() needed!

# Option 2: Backprop mode (drop-in SGD replacement)
optimizer = muon_backprop(model.parameters())
loss.backward()
optimizer.step()
```

### Optimal EP Configuration

For classification tasks:

```python
from mep import smep

optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    beta=0.5,           # Nudging strength
    settle_steps=30,    # Settling iterations
    settle_lr=0.15,     # Settling learning rate
    loss_type='mse',    # Stable energy computation
    use_error_feedback=False,
)
```

See [benchmarks/PERFORMANCE_BASELINES.md](benchmarks/PERFORMANCE_BASELINES.md) for full configuration guide.

---

## 📊 Performance Summary

| Benchmark | EP | SGD | Adam | Status |
|-----------|-----|-----|------|--------|
| MNIST (3 epoch) | **91.4%** | 91.0% | 90.2% | ✅ EP wins |
| MNIST (10 epoch) | 95.37% | 93.80% | **95.75%** | ✅ EP ties Adam |
| XOR (100 step) | 100% | 100% | 100% | ✅ Parity |

**Key Findings:**
- EP achieves performance parity with backpropagation
- EP is ~2× slower (fundamental algorithmic cost)
- EP uses more memory than backprop+checkpointing
- Dropout is incompatible with EP settling

Full results: [benchmarks/VALIDATION_RESULTS.md](benchmarks/VALIDATION_RESULTS.md)

---

## 🎯 When to Use EP

### Use EP For:
- ✅ Biological plausibility research
- ✅ Neuromorphic hardware deployment
- ✅ Energy-based model research
- ✅ Educational demonstrations
- ✅ Studying alternative learning mechanisms

### Use Backprop For:
- ✅ Standard classification/regression
- ✅ Production training pipelines
- ✅ Speed-critical applications
- ✅ Maximum accuracy goals

See [research/ROADMAP_RESEARCH.md](research/ROADMAP_RESEARCH.md) for detailed guidance.

---

## 🔧 API Reference

### Preset Optimizers

| Function | Description | Use Case |
|----------|-------------|----------|
| `smep()` | Spectral Muon EP | Standard EP with Muon orthogonalization |
| `sdmep()` | Spectral Dion-Muon EP | Large models (Dion low-rank updates) |
| `local_ep()` | Local EP | Layer-local updates (biologically plausible) |
| `natural_ep()` | Natural EP | Fisher Information whitening |
| `muon_backprop()` | Muon + Backprop | Drop-in SGD replacement |

### Strategy Classes

| Class | Purpose |
|-------|---------|
| `EPGradient` | Free/nudged phase contrast |
| `LocalEPGradient` | Layer-local EP gradients |
| `NaturalGradient` | Fisher Information gradient |
| `MuonUpdate` | Newton-Schulz orthogonalization |
| `DionUpdate` | Low-rank SVD updates |
| `SpectralConstraint` | Spectral norm constraints |
| `ErrorFeedback` | Residual accumulation |

See source code in [mep/optimizers/](../mep/optimizers/) for full API.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run performance regression tests
pytest tests/regression/test_performance_baseline.py -v

# Run with coverage
pytest tests/ --cov=mep --cov-report=html
```

---

## 🤝 Contributing

See [research/ROADMAP_RESEARCH.md](research/ROADMAP_RESEARCH.md) for:
- Current research priorities
- Collaboration opportunities
- How to contribute

---

## 📄 License

MIT License. See [LICENSE](../LICENSE) for details.

---

*Last updated: 2026-02-18*
