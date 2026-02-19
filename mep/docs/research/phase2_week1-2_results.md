# Phase 2: Week 1-2 Results

**Date:** 2026-02-18
**Status:** ✅ Correctness Verified, ⚠️ Memory Savings Not Yet Achieved

---

## Executive Summary

Week 1-2 of Phase 2 focused on memory profiling and O(1) prototype implementation. Key achievements:

1. ✅ **Baseline memory curve established** - EP and backprop scale identically (0.1326 MB/layer)
2. ✅ **Component breakdown completed** - Settling (32%), Energy (32%), Contrast (36%)
3. ✅ **PyTorch operations identified** - `enable_grad()` adds 1.62 MB overhead
4. ✅ **Manual settling implemented** - Produces identical states (<1e-5 difference)
5. ✅ **Manual energy implemented** - Produces identical values (<1e-8 difference)
6. ⚠️ **Memory savings not achieved** - O(1) prototype uses similar memory to current EP

---

## Baseline Measurements

### Memory vs Depth Scaling

| Depth | Backprop Activation (MB) | EP Activation (MB) | EP Time (s) |
|-------|-------------------------|-------------------|-------------|
| 10    | 20.93                   | 20.93             | 0.24        |
| 50    | 25.79                   | 25.78             | 0.71        |
| 100   | 32.44                   | 32.44             | 1.41        |
| 200   | 45.75                   | 45.75             | 2.92        |
| 500   | 85.69                   | 85.69             | 7.20        |
| 1000  | 152.25                  | 152.25            | 14.38       |

**Scaling:** 0.1326 MB/layer for both methods

**Key finding:** Current EP implementation stores activations just like backpropagation.

---

## Component Memory Profile (Depth 100)

| Component  | Memory (MB) | Time (ms) | Fraction |
|------------|-------------|-----------|----------|
| Settling   | 34.84       | 2812      | 32.2%    |
| Energy     | 34.82       | 61        | 32.2%    |
| Contrast   | 38.52       | 64        | 35.6%    |
| **Total**  | **108.19**  | **2937**  | **100%** |

**Key finding:** All three components contribute equally to memory usage.

---

## PyTorch Operation Analysis

| Operation                | Memory (MB) | Overhead |
|-------------------------|-------------|----------|
| Manual (no_grad)        | 25.07       | baseline |
| nn.Module (no_grad)     | 25.07       | 0.00 MB  |
| nn.Module (enable_grad) | 26.69       | 1.62 MB  |

**Key finding:** `enable_grad()` triggers activation storage, adding 1.62 MB overhead.

---

## O(1) Prototype Correctness

### Test Results

| Test                          | Target      | Actual      | Status |
|-------------------------------|-------------|-------------|--------|
| Manual energy difference      | < 1e-8      | 0.0         | ✅ PASS |
| Manual settling difference    | < 1e-5      | 0.0         | ✅ PASS |
| Training loss difference      | < 1e-3      | 4e-6        | ✅ PASS |

**Key finding:** O(1) prototype produces mathematically identical results.

---

## Memory Savings: O(1) Prototype vs Current EP

| Depth | Current EP (MB) | O(1) EP (MB) | Savings |
|-------|----------------|--------------|---------|
| 10    | 37.35          | 37.58        | 0.0%    |
| 50    | 44.85          | 47.71        | 0.0%    |
| 100   | 54.81          | 60.89        | 0.0%    |
| 200   | 74.72          | 87.26        | 0.0%    |
| 500   | 134.47         | 166.36       | 0.0%    |

**Status:** ⚠️ No memory savings achieved. O(1) prototype uses 10-20% MORE memory.

**Root cause:** The current implementation still builds a computation graph during settling when computing state gradients. While we don't store the settling **history**, we still store O(depth) activations for each settling step.

---

## Success Criteria Assessment

| Criterion                           | Target | Actual | Status |
|-------------------------------------|--------|--------|--------|
| Baseline memory curve               | ✅ Complete | ✅ Complete | ✅ |
| Component breakdown                 | ✅ Complete | ✅ Complete | ✅ |
| PyTorch operation analysis          | ✅ Complete | ✅ Complete | ✅ |
| Manual settling correctness         | < 1e-5 | 0.0 | ✅ |
| Manual energy correctness           | < 1e-8 | 0.0 | ✅ |
| Initial memory savings at depth 500 | 50%+   | 0% | ❌ |

**Overall:** 5/6 criteria met. Memory savings requires additional work.

---

## Technical Analysis: Why Memory Savings Not Achieved

### Current Approach (Still Stores Activations)

```python
# In settle_manual():
for step in range(steps):
    # Create states that require grad
    states_for_grad = [s.detach().clone().requires_grad_(True) for s in states]
    
    # Compute energy WITH grad tracking
    E = manual_energy_compute(..., use_grad=True)
    
    # This triggers autograd to store O(depth) activations
    grads = torch.autograd.grad(E, states_for_grad)
```

**Problem:** Even though we use `manual_energy_compute`, setting `use_grad=True` causes PyTorch to store all intermediate activations for the backward pass.

### Required Approach (True O(1))

To achieve O(1) memory, we need to compute `dE/dstate` **without** building the forward graph:

1. **Analytic gradients:** Derive and implement `dE/dstate` formula directly
2. **Finite differences:** Approximate gradients numerically (slow but O(1) memory)
3. **Custom CUDA kernel:** Implement settling kernel that computes gradients in-place

**Recommended:** Option 3 - custom CUDA kernel for fused settling + gradient computation.

---

## Files Created

### Profiling Scripts
- `examples/profile_memory_detailed.py` - Comprehensive memory profiling
- `examples/verify_o1_memory.py` - O(1) prototype verification

### O(1) Implementation
- `mep/optimizers/o1_memory.py` - Manual settling and energy computation

### Documentation
- `docs/research/phase2_week1-2_results.md` - This file

---

## Next Steps (Week 3-4)

### Immediate Actions

1. **Implement analytic state gradients**
   - Derive formula for `dE/dstate` for each layer type
   - Implement direct gradient computation without autograd

2. **Profile gradient computation**
   - Identify which operations store activations during `torch.autograd.grad()`
   - Test gradient checkpointing for state gradients

3. **Custom CUDA kernel design**
   - Specify kernel interface for fused settling
   - Implement manual gradient computation in CUDA

### Revised Timeline

| Week | Deliverable | Success Metric |
|------|-------------|----------------|
| 3    | Analytic gradients | Memory reduced 30%+ |
| 4    | Gradient checkpointing | Memory reduced 50%+ |
| 5-6  | CUDA kernel | Memory flat vs depth |

---

## Lessons Learned

1. **`torch.no_grad()` is not enough** - Even with no_grad context, calling `autograd.grad()` on tensors that require grad triggers activation storage.

2. **The settling loop is the problem** - 30 settling steps × O(depth) activations = O(steps × depth) memory.

3. **Correctness is achievable** - Manual implementations can match current EP exactly.

4. **True O(1) requires avoiding autograd entirely** - During settling, we need direct gradient computation, not autograd.

---

## Conclusion

Week 1-2 successfully established baselines and verified correctness of manual implementations. However, the key goal of 50%+ memory savings at depth 500 was not achieved.

**Root cause:** The O(1) prototype still relies on `torch.autograd.grad()` for state gradients, which stores activations.

**Path forward:** Implement analytic state gradients or custom CUDA kernel to avoid autograd entirely during settling.

**Risk:** If true O(1) memory cannot be achieved with standard PyTorch, custom CUDA development will be required (additional 2-4 weeks).

---

*Created: 2026-02-18*
*Status: Week 1-2 Complete, Week 3-4 Planning*
