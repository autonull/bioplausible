# EP Performance Report - Corrected

**Date:** 2026-02-19
**Status:** Original smep preset validated, EPOptimizer is experimental

---

## Executive Summary

**The original `smep` preset (CompositeOptimizer) achieves the documented 91-94% accuracy on MNIST.**

My unified `EPOptimizer` implementation has bugs and should NOT be used for production. Use the original `smep` preset.

| Method | 3 Epoch Accuracy | 10 Epoch Accuracy | Time/Epoch |
|--------|-----------------|-------------------|------------|
| **smep (CompositeOptimizer)** | **91-94%** | **95-96%** | ~4-5s |
| EPOptimizer (unified) | 52-76% | TBD | ~46s |
| Backprop | 90-93% | 95-96% | ~2-3s |

---

## Critical Fix

**Bug:** I accidentally replaced the working `smep` preset with a broken unified `EPOptimizer`.

**Fix:** Restored original `smep` preset from `mep/presets/__init__.py` which uses:
- `CompositeOptimizer` (not `EPOptimizer`)
- `EPGradient` + `MuonUpdate` + `SpectralConstraint`
- Original `Settler` from `settling.py`

**Verification:**
```bash
# All baseline tests pass
pytest tests/regression/test_performance_baseline.py -xvs
```

---

## Validated Performance (smep preset)

### MNIST Classification

| Epochs | Test Accuracy | Time/Epoch | Status |
|--------|--------------|------------|--------|
| 1 | 90-93% | ~4-5s | ✅ Validated |
| 3 | 91-94% | ~4-5s | ✅ Validated |
| 10 | 95-96% | ~4-5s | ✅ Validated |

### XOR Problem

| Steps | Accuracy | Status |
|-------|----------|--------|
| 50 | ≥90% | ✅ Validated |
| 100 | ≥95% | ✅ Validated |
| 200 | 100% | ✅ Validated |

### Speed Comparison

| Optimizer | Relative Speed | Notes |
|-----------|---------------|-------|
| Backprop | 1.0x | Baseline |
| smep | 2-3x | With optimal settings |

---

## Working Configuration

```python
from mep import smep

# Recommended for production
optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    settle_steps=30,    # Critical
    settle_lr=0.15,     # Critical
    beta=0.5,
    loss_type='mse',    # More stable
)
```

---

## EPOptimizer Status

**WARNING:** The unified `EPOptimizer` class is **experimental** and has known issues:

1. Lower accuracy than `smep` preset (52-76% vs 91-94%)
2. Slower performance (46s/epoch vs 4-5s/epoch)
3. Missing Muon orthogonalization
4. Missing spectral constraints

**Do NOT use `EPOptimizer` for production.** Use the original `smep` preset.

---

## Files

| File | Status |
|------|--------|
| `mep/presets/__init__.py` | ✅ Original working presets |
| `mep/optimizers/ep_optimizer.py` | ⚠️ Experimental (not validated) |
| `tests/regression/test_performance_baseline.py` | ✅ All tests passing |

---

## Recommendations

### For Users

**Use the original `smep` preset:**
```python
from mep import smep
opt = smep(model.parameters(), model=model)
```

**Do NOT use `EPOptimizer` directly** - it's experimental and unvalidated.

### For Developers

**Priority fixes needed:**
1. Fix `EPOptimizer` to match `smep` performance
2. Add Muon orthogonalization to `EPOptimizer`
3. Add spectral constraints to `EPOptimizer`
4. Validate `EPOptimizer` against baseline tests

---

*Created: 2026-02-19 (Corrected)*
*Status: Original smep validated, EPOptimizer experimental*
