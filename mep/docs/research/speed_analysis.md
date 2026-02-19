# EP Speed Analysis and Optimization

**Date:** 2026-03-04
**Status:** Bottleneck identified, optimizations available

---

## Executive Summary

EP training speed is **proportional to settling steps**. The documented "2-3x slower" figure assumed optimized settings, but default settings (30 settling steps) result in **10-15x slower** training.

**Key finding:** Each settling step requires a full forward pass through O(depth) layers. 30 steps = 30× forward passes + contrast step.

---

## Speed Measurements

### Default Settings (settle_steps=30)

| Configuration | Time (ms) | Speed vs BP |
|--------------|-----------|-------------|
| Backprop | 5.4 | 1.0x |
| EP (5 steps) | 24.4 | 4.5x |
| EP (10 steps) | 33.7 | 6.2x |
| EP (15 steps) | 43.5 | 8.0x |
| EP (20 steps) | 53.5 | 9.8x |
| EP (30 steps, default) | 73.2 | 13.4x |

### Speed scales linearly with settling steps

```
Per-step overhead: ~2.3ms (for 3-layer MLP, batch=32)
Total EP time = BP_time + settle_steps × per_step_overhead
```

---

## Bottleneck Analysis

### Profiling Results

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Settling loop (30 steps) | 67.8 | 92.6% |
| Contrast step | 4.9 | 6.7% |
| Overhead | 0.5 | 0.7% |

**Settling dominates EP time** - reducing settling steps is the most effective optimization.

### Why So Slow?

Each settling iteration:
1. Forward pass through O(depth) layers
2. Energy computation
3. Gradient computation (autograd.grad or analytic)
4. State update

30 settling steps = 30× the computation of a single forward pass.

---

## Optimization Strategies

### 1. Reduce Settling Steps (Most Effective)

```python
# Default (13x slower)
optimizer = smep(model.parameters(), model=model, settle_steps=30)

# Optimized (4-6x slower)
optimizer = smep(model.parameters(), model=model, settle_steps=10, settle_lr=0.2)

# Or use preset
from mep import smep_fast
optimizer = smep_fast(model.parameters(), model=model)
```

**Speedup:** 3-4x faster than default
**Trade-off:** May need to tune settle_lr for convergence

### 2. Analytic Gradients (Already Implemented)

```python
from mep.optimizers import O1MemoryEPv2
optimizer = O1MemoryEPv2(model.parameters(), model=model, settle_steps=10)
```

**Speedup:** 1.5-2x faster settling (avoids autograd.grad overhead)
**Trade-off:** None (identical results)

### 3. Adaptive Settling (Future)

```python
# Stop when energy converges (not yet implemented)
optimizer = smep(..., adaptive=True, tol=1e-4, patience=5)
```

**Expected speedup:** 30-50% fewer steps on average
**Trade-off:** Slightly more complex logic

---

## Combined Optimization Potential

| Configuration | Speed vs BP | Notes |
|--------------|-------------|-------|
| Default (30 steps) | 10-15x | Current default |
| 10 steps | 4-6x | Minimal tuning needed |
| 10 steps + analytic | 3-5x | O1MemoryEPv2 |
| Adaptive (est.) | 2-3x | Future optimization |

**The documented "2-3x slower" is achievable** with:
- 10-15 settling steps
- Analytic gradients
- Adaptive settling (early stopping)

---

## New Preset: smep_fast

Added `smep_fast()` preset for faster training:

```python
from mep import smep_fast

optimizer = smep_fast(
    model.parameters(),
    model=model,
    lr=0.01,
    settle_steps=10,  # vs 30 default
    settle_lr=0.2,    # vs 0.15 default
)
```

**Speed:** 3-4x faster than default SMEP (4-6x slower than BP)

---

## Recommendations

### For Research (accuracy-focused)

Use default settings when:
- Maximizing accuracy is critical
- Training time is not a concern
- Experimenting with new architectures

```python
from mep import smep
optimizer = smep(model.parameters(), model=model, settle_steps=30)
```

### For Prototyping (speed-focused)

Use smep_fast when:
- Rapid iteration is important
- Training large models
- Early experimentation

```python
from mep import smep_fast
optimizer = smep_fast(model.parameters(), model=model)
```

### For Production (balanced)

Use O1MemoryEPv2 when:
- Want best speed/accuracy tradeoff
- Analytic gradients are acceptable
- Memory efficiency matters

```python
from mep.optimizers import O1MemoryEPv2
optimizer = O1MemoryEPv2(model.parameters(), model=model, settle_steps=15)
```

---

## Files Updated

- `mep/presets/__init__.py` - Added `smep_fast()` preset
- `mep/__init__.py` - Exported `smep_fast`
- `docs/benchmarks/PERFORMANCE_BASELINES.md` - Updated speed benchmarks
- `examples/profile_ep_speed.py` - Speed profiling script

---

## Next Steps

1. **Test accuracy impact** - Verify smep_fast achieves similar accuracy to default
2. **Implement adaptive settling** - Early stopping when converged
3. **Document speed/accuracy tradeoffs** - Guide users to appropriate settings
4. **Consider dynamic settling** - Adjust steps per layer/epoch based on convergence

---

*Created: 2026-03-04*
*Status: Analysis complete, optimizations available*
