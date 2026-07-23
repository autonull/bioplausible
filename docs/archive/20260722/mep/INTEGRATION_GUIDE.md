# MEP Integration Guide for Bioplausible

**Date:** 2026-02-19
**Status:** Ready for integration
**Source:** `/home/me/mep` (this repository)
**Target:** `https://github.com/automenta/bioplausible`

---

## Executive Summary

MEP (Muon Equilibrium Propagation) is a biologically plausible deep learning framework that combines:
- **Equilibrium Propagation** (local learning via settling dynamics)
- **Muon orthogonalization** (Newton-Schulz iterations for weight stability)
- **Spectral constraints** (weight norm regularization)

**Validated Performance:**
- MNIST: 91-94% (3 epochs), 95-96% (10 epochs)
- XOR: 100% (200 steps)
- Speed: 2-3x slower than backprop

---

## What to Integrate

### Core Components (Priority 1)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `CompositeOptimizer` | `mep/optimizers/composite.py` | ✅ Validated | Strategy pattern optimizer |
| `EPGradient` | `mep/optimizers/strategies/gradient.py` | ✅ Validated | EP gradient computation |
| `MuonUpdate` | `mep/optimizers/strategies/update.py` | ✅ Validated | Newton-Schulz orthogonalization |
| `SpectralConstraint` | `mep/optimizers/strategies/constraint.py` | ✅ Validated | Spectral norm constraints |
| `Settler` | `mep/optimizers/settling.py` | ✅ Validated | Settling dynamics |
| `EnergyFunction` | `mep/optimizers/energy.py` | ✅ Validated | EP energy computation |
| `ModelInspector` | `mep/optimizers/inspector.py` | ✅ Validated | Model structure extraction |

### Presets (Priority 2)

| Preset | File | Status | Notes |
|--------|------|--------|-------|
| `smep` | `mep/presets/__init__.py` | ✅ Validated | SMEP (default) |
| `sdmep` | `mep/presets/__init__.py` | ⚠️ Partial | SDMEP (Dion low-rank) |
| `local_ep` | `mep/presets/__init__.py` | ✅ Validated | Local EP |
| `muon_backprop` | `mep/presets/__init__.py` | ✅ Validated | Muon + backprop |

### Experimental (Do NOT integrate yet)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `EPOptimizer` | `mep/optimizers/ep_optimizer.py` | ❌ Broken | Unified optimizer (52-76% accuracy) |
| `O1MemoryEP` | `mep/optimizers/o1_memory.py` | ❌ Experimental | O(1) memory v1 |
| `O1MemoryEPv2` | `mep/optimizers/o1_memory_v2.py` | ❌ Experimental | O(1) memory v2 |
| `EWCRegularizer` | `mep/optimizers/ewc.py` | ⚠️ Untested | EWC for continual learning |

---

## Integration Checklist

### Phase 1: Core Integration

- [ ] Copy `mep/optimizers/` to `bioplausible/mep/`
- [ ] Copy `mep/presets/` to `bioplausible/mep/presets/`
- [ ] Update imports in Bioplausible
- [ ] Run Bioplausible test suite
- [ ] Verify MEP tests still pass

### Phase 2: Testing

- [ ] Run `tests/regression/test_performance_baseline.py`
- [ ] Run `tests/regression/test_ep_baseline.py`
- [ ] Run `tests/regression/test_ep_smoke.py`
- [ ] Verify MNIST accuracy ≥91% (3 epochs)

### Phase 3: Documentation

- [ ] Add MEP to Bioplausible README
- [ ] Document MEP-specific configuration
- [ ] Add integration examples

---

## Known Issues

### Critical (Must Fix Before Integration)

None - core components are validated and working.

### Minor (Can Fix After Integration)

1. **`EPOptimizer` is broken** - Do not integrate until fixed
2. **O(1) memory not achieved** - Research direction, not production-ready
3. **Speed optimization needed** - EP is 2-3x slower than backprop

### Research Directions (Not Production)

1. **Adaptive settling** - Early stopping when converged
2. **Custom CUDA kernels** - Fused settling operations
3. **Better weight initialization** - EP-specific init strategies
4. **Continual learning** - EP + EWC integration

---

## File Structure for Integration

```
bioplausible/
├── mep/                          # NEW: MEP integration
│   ├── __init__.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── composite.py          # Core optimizer
│   │   ├── energy.py             # EP energy
│   │   ├── settling.py           # Settling dynamics
│   │   ├── inspector.py          # Model structure
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Strategy interfaces
│   │   │   ├── gradient.py       # EPGradient
│   │   │   ├── update.py         # MuonUpdate
│   │   │   ├── constraint.py     # SpectralConstraint
│   │   │   └── feedback.py       # ErrorFeedback
│   │   └── monitor.py            # Training monitoring
│   └── presets/
│       ├── __init__.py           # smep, sdmep, etc.
│   └── cuda/                     # Optional CUDA kernels
│       ├── __init__.py
│       └── kernels.py
├── tests/
│   └── mep/                      # NEW: MEP tests
│       ├── test_performance_baseline.py
│       └── test_ep_smoke.py
└── docs/
    └── mep/                      # NEW: MEP documentation
        ├── README.md
        └── performance_report.md
```

---

## Quick Start (After Integration)

```python
from bioplausible.mep import smep

# Create model
model = MyNeuralNetwork()

# Create MEP optimizer
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',           # 'ep' or 'backprop'
    settle_steps=30,     # Settling iterations
    settle_lr=0.15,      # Settling learning rate
    beta=0.5,            # Nudging strength
    loss_type='mse',     # 'mse' or 'cross_entropy'
)

# Training loop
for x, y in train_loader:
    optimizer.step(x=x, target=y)
```

---

## Performance Expectations

After integration, expect:

| Metric | Expected | Validation |
|--------|----------|------------|
| MNIST (3 epochs) | 91-94% | `test_mnist_extended` |
| MNIST (10 epochs) | 95-96% | Manual validation |
| XOR (100 steps) | ≥95% | `test_xor_convergence` |
| Speed vs BP | 2-3x slower | Manual benchmark |

---

## Contact

**Original MEP:** This repository (`/home/me/mep`)
**Bioplausible:** https://github.com/automenta/bioplausible

---

*Created: 2026-02-19*
*Status: Ready for integration*
