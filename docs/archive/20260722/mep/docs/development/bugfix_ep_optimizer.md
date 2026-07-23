# Bug Fix: EP Unified Optimizer

**Date:** 2026-02-19
**Bug:** EP not learning (accuracy stuck at random baseline)
**Fix:** Two critical bugs in `mep/optimizers/ep_optimizer.py`

---

## Bug #1: States Missing `requires_grad=True`

### Symptom
Settling loop didn't change states at all (diff = 0.000000).

### Root Cause
`_capture_states()` was capturing states without `requires_grad=True`:
```python
# BROKEN
states.append(o.detach().float().clone())
```

### Fix
```python
# FIXED
states.append(o.detach().float().clone().requires_grad_(True))
```

### Why It Matters
The settling loop computes gradients w.r.t. states. If states don't have `requires_grad=True`, autograd can't compute gradients and the settling doesn't move.

---

## Bug #2: Analytic Gradients Too Simplistic

### Symptom
Settling only changed output layer, not hidden layers.

### Root Cause
`_analytic_gradients()` computed per-layer gradients independently:
```python
# BROKEN - doesn't account for inter-layer dependencies
grad = (state - h) / batch_size
```

This ignores the fact that changing one layer's state affects what earlier layers "should" produce.

### Fix
Use autograd for settling gradients:
```python
# FIXED - default to autograd
gradient_method='autograd'  # in EPConfig
```

### Why It Matters
EP settling requires FULL gradients through the energy function, including indirect effects through the network. Simple per-layer MSE gradients don't capture this.

---

## Verification

### Smoke Test (< 20 seconds)
```bash
python tests/regression/test_ep_smoke.py
```
**Pass criteria:** >15% MNIST accuracy (1 epoch, 10k samples)

### Full Regression (< 3 minutes)
```bash
python tests/regression/test_ep_baseline.py
```
**Pass criteria:** >55% MNIST accuracy (3 epochs, full data)

### Current Status
- ✅ Smoke test: 22.5% accuracy, 15s
- ✅ Full regression: 71% accuracy, 150s

---

## Lessons Learned

1. **Always test with known working config first** - Should have compared against `smep` preset immediately

2. **EP settling requires autograd** - The analytic gradient approximation is too simplistic for the settling dynamics

3. **Two-tier testing is essential**:
   - Smoke test (<20s): Catches fundamental bugs
   - Full regression (<3min): Verifies actual learning

4. **The original `smep` preset works** - Use it as the reference when debugging

---

## Files Changed

| File | Change |
|------|--------|
| `mep/optimizers/ep_optimizer.py` | Fixed `_capture_states()` to set `requires_grad=True`; Default to `gradient_method='autograd'` |
| `tests/regression/test_ep_smoke.py` | Created smoke test suite |
| `docs/development/workflow.md` | Updated with two-tier testing workflow |

---

*Created: 2026-02-19*
*Status: Fixed and Verified*
