# EP Performance Report

**Date:** 2026-02-19
**Status:** EP working, but accuracy gap vs backprop remains

---

## Executive Summary

EP is **functional and stable** across all configurations, but achieves lower accuracy than backpropagation on MNIST:

| Method | 3 Epoch Accuracy | Time/Epoch | Speed |
|--------|-----------------|------------|-------|
| **EP (30 settle steps)** | 52-69% | 46s | 1x |
| **EP (10 settle steps)** | 30-50% | 23s | 2x |
| **Backprop** | 95-97% | 10s | 4.6x |

**Key Finding:** EP accuracy is highly sensitive to hyperparameters. The `smep` preset achieves 68% after 3 epochs with proper configuration.

---

## Benchmark Results

### Configuration Comparison (3 epochs, MNIST)

| Config | Settle Steps | Test Acc | Time/Epoch | Samples/sec |
|--------|-------------|----------|------------|-------------|
| EP_fast | 5 | 27% | 17s | 3561 |
| EP_default | 10 | 51% | 23s | 2636 |
| EP_accurate | 30 | 52% | 46s | 1311 |
| EP_deep (10 layer) | 10 | 22% | 50s | 1202 |
| **BP_default** | - | **97%** | **10s** | **6167** |
| BP_deep (10 layer) | - | 95% | 10s | 5734 |

### Working Configuration (Verified)

```python
from mep import smep

optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode="ep",
    settle_steps=30,  # Critical for accuracy
    settle_lr=0.15,  # Critical for convergence
    beta=0.5,
    loss_type="mse",  # More stable than cross_entropy
)
```

**Expected:** 55% → 64% → 68% over 3 epochs

---

## Performance Analysis

### Speed Breakdown

| Component | EP Time | BP Time | Ratio |
|-----------|---------|---------|-------|
| Forward pass | ~2s | ~2s | 1x |
| Settling (30 steps) | ~40s | - | - |
| Contrast step | ~4s | ~8s (backward) | 0.5x |
| **Total** | **46s** | **10s** | **4.6x** |

**Bottleneck:** Settling loop (87% of EP time)

### Accuracy Gap Analysis

| Factor | Impact | Notes |
|--------|--------|-------|
| Settling convergence | High | Insufficient settling → poor accuracy |
| Learning rate | Medium | EP needs lower LR than BP |
| Loss type | High | MSE works better than cross_entropy |
| Architecture | Medium | 256→128 works better than 256→256 |

---

## Recommendations

### For Users

**Best accuracy:**
```python
from mep import smep

opt = smep(model.parameters(), model=model, settle_steps=30)
```

**Best speed/accuracy tradeoff:**
```python
from mep import smep

opt = smep(model.parameters(), model=model, settle_steps=20, settle_lr=0.2)
```

**Fastest (prototyping):**
```python
from mep import smep_fast

opt = smep_fast(model.parameters(), model=model)
```

### For Developers

**Optimization opportunities:**

1. **Reduce settling steps** - Currently 30, could be 15-20 with adaptive stopping
2. **Fuse settling operations** - Current implementation has Python overhead
3. **Better initialization** - EP may need different weight init than BP
4. **Learning rate tuning** - EP optimal LR differs from BP

---

## Known Limitations

1. **Speed:** EP is 4-5x slower than backprop (fundamental settling cost)
2. **Accuracy gap:** EP achieves 50-70% vs BP's 95%+ (3 epochs, MNIST)
3. **Hyperparameter sensitivity:** EP requires more tuning than BP
4. **Deep networks:** EP accuracy degrades with depth (>10 layers)

---

## Files

- `benchmarks/performance_suite.py` - Performance benchmark suite
- `tests/regression/test_ep_smoke.py` - Smoke test (<20s)
- `tests/regression/test_ep_baseline.py` - Full regression (<3min)

---

*Created: 2026-02-19*
*Next: Hyperparameter optimization for better accuracy*
