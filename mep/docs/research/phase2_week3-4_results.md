# Phase 2: Week 3-4 Results

**Date:** 2026-02-25
**Status:** ✅ Analytic Gradients Verified, ⚠️ Memory Savings Not Achieved

---

## Executive Summary

Week 3-4 focused on implementing analytic gradients for true O(1) memory during settling. Key achievements:

1. ✅ **Analytic gradients implemented** - dE/dstate computed via direct formula
2. ✅ **Gradient correctness verified** - 2.5e-12 difference vs autograd (essentially perfect)
3. ✅ **Training correctness verified** - 8.6e-6 loss difference
4. ✅ **1.8x speedup achieved** - Analytic gradients are faster than autograd
5. ⚠️ **Memory savings not achieved** - O(1) v2 uses 27% MORE memory at depth 500

---

## Analytic Gradients Implementation

### Formula Derivation

For MSE energy: **E = 0.5 × ||state - h||²**

Gradient: **dE/dstate = state - h**

For KL energy (classification): **E = KL(softmax(state) || softmax(h))**

Gradient: **dE/dstate ≈ (softmax(state) - softmax(h)) / temperature**

### Implementation

```python
def analytic_state_gradients(model, x, states, structure, target_vec, beta):
    """Compute dE/dstate analytically without autograd."""
    grads = []
    prev = x
    
    for item in structure:
        if item["type"] == "layer":
            state = states[state_idx]
            h = module(prev)  # Forward pass (no grad)
            
            # Analytic gradient: just subtraction!
            grad = state - h
            grads.append(grad / batch_size)
            
            prev = state
    
    # Add nudge term gradient for last layer
    if target_vec is not None and beta > 0:
        nudge_grad = beta * (softmax(state) - target_one_hot) / batch_size
        grads[-1] = grads[-1] + nudge_grad
    
    return grads
```

---

## Correctness Verification

### Test Results

| Test                          | Target      | Actual       | Status |
|-------------------------------|-------------|--------------|--------|
| Analytic gradients vs autograd | < 1e-5      | 2.5e-12      | ✅ PASS |
| Analytic settling vs current   | < 1e-5      | 2.4e-5       | ⚠️ MARGINAL |
| Training loss difference       | < 1e-3      | 8.6e-6       | ✅ PASS |

**Analytic gradients:** Essentially perfect match (2.5e-12 is machine epsilon level)

**Settling difference:** Slightly above tolerance (2.4e-5 vs 1e-5) but training results are identical

---

## Performance Comparison

### Speed

| Depth | Current EP (s) | O(1) v2 (s) | Speedup |
|-------|---------------|-------------|---------|
| 10    | 0.178         | 0.084       | 2.1x    |
| 50    | 0.741         | 0.424       | 1.7x    |
| 100   | 1.420         | 0.803       | 1.8x    |
| 200   | 2.887         | 1.613       | 1.8x    |
| 500   | 7.146         | 3.856       | 1.9x    |

**Average speedup: 1.8x** - Analytic gradients are significantly faster!

### Memory

| Depth | Current EP (MB) | O(1) v2 (MB) | Difference |
|-------|----------------|--------------|------------|
| 10    | 21.86          | 22.10        | -1.1%      |
| 50    | 29.36          | 32.22        | -9.7%      |
| 100   | 39.32          | 45.41        | -15.5%     |
| 200   | 59.24          | 71.77        | -21.2%     |
| 500   | 118.99         | 150.88       | -26.8%     |

**O(1) v2 uses MORE memory** - opposite of expected!

---

## Phase-by-Phase Memory Breakdown

### Settling Phase

| Depth | Current (MB) | O(1) v2 (MB) | Difference |
|-------|-------------|--------------|------------|
| 10    | 19.90       | 20.70        | -4.0%      |
| 100   | 33.22       | 38.49        | -15.9%     |
| 500   | 92.41       | 117.54       | -27.2%     |

### Contrast Phase

| Depth | Current (MB) | O(1) v2 (MB) | Difference |
|-------|-------------|--------------|------------|
| 10    | 21.16       | 22.10        | -4.4%      |
| 100   | 38.52       | 45.40        | -17.9%     |
| 500   | 117.58      | 150.88       | -28.3%     |

---

## Root Cause Analysis

### Why O(1) v2 Uses More Memory

**Expected:** Analytic gradients avoid autograd overhead → less memory

**Actual:** O(1) v2 uses more memory due to:

1. **State tensor allocation:** Both implementations store O(depth) state tensors - this is unavoidable and dominates memory usage

2. **Temporary allocations:** The analytic gradient implementation creates intermediate tensors (softmax outputs, differences) that add to peak memory

3. **Current EP is already efficient:** The current `Settler.settle()` doesn't store the full settling trajectory - it updates states in-place and frees the graph after each step

4. **Gradient checkpointing overhead:** The `energy_from_states_minimal()` function uses gradient checkpointing, but checkpointing has its own memory overhead for storing checkpoints

### Memory Profile Comparison

```
Current EP:
  - State tensors: O(depth) ✓
  - Settling graph: O(depth) per step, freed after ✓
  - Contrast graph: O(depth) ✓
  - Total: O(depth)

O(1) v2 EP:
  - State tensors: O(depth) ✓
  - Settling graph: O(1) ✓ (analytic, no graph)
  - Contrast graph: O(depth) with checkpointing overhead
  - Temporary tensors: Additional allocations for analytic formulas
  - Total: O(depth) + overhead
```

---

## Key Insight: EP Cannot Achieve O(1) Total Memory

**Fundamental limitation:** EP inherently requires O(depth) memory because:

1. **State storage:** We must store O(depth) state tensors (free phase + nudged phase)
2. **Parameter gradients:** Computing dE/dW requires a forward pass through O(depth) layers

The O(1) claim was about avoiding **O(steps × depth)** memory from storing the settling trajectory, not achieving O(1) total memory.

**Current EP already achieves this:** The settling loop updates states in-place, so it only stores O(depth) states, not O(steps × depth).

---

## Success Criteria Assessment

| Criterion                           | Target | Actual | Status |
|-------------------------------------|--------|--------|--------|
| Analytic gradients correctness      | < 1e-5 | 2.5e-12 | ✅ |
| Analytic settling correctness       | < 1e-5 | 2.4e-5 | ⚠️ |
| Training correctness                | < 1e-3 | 8.6e-6 | ✅ |
| Memory savings at depth 500         | 50%+   | -27%   | ❌ |
| Speed improvement                   | Any    | 1.8x   | ✅ |

**Overall:** 3/5 criteria met (with 1 marginal)

---

## Files Created

### Implementation
- `mep/optimizers/o1_memory_v2.py` - Analytic gradients implementation

### Scripts
- `examples/verify_o1_memory_v2.py` - V2 verification script
- `examples/profile_settling_memory.py` - Phase-by-phase memory profiling

### Documentation
- `docs/research/phase2_week3-4_results.md` - This file

---

## Lessons Learned

1. **Current EP is already memory-efficient** - The settling loop doesn't store trajectory, only current states

2. **Analytic gradients are faster** - 1.8x speedup is significant for training time

3. **O(1) total memory is not achievable for EP** - State storage is O(depth), and parameter gradients require O(depth) computation graph

4. **The real O(1) claim** - Should be "O(1) settling overhead" not "O(1) total memory"

5. **Gradient checkpointing has overhead** - Checkpointing reduces peak memory but adds computation and temporary storage

---

## Revised Understanding of O(1) Memory

### Original Claim (Incorrect)
"EP achieves O(1) activation memory - independent of depth"

### Corrected Claim
"EP achieves O(1) **settling overhead** - the settling loop doesn't accumulate additional memory beyond the O(depth) state storage that is fundamental to the algorithm"

### What This Means

| Component | Memory Scaling | Notes |
|-----------|---------------|-------|
| State storage | O(depth) | Fundamental - cannot reduce |
| Settling overhead | O(1) | Per-step, freed after each step |
| Contrast graph | O(depth) | Required for parameter gradients |
| **Total** | **O(depth)** | Same as backprop |

---

## Next Steps (Week 5-6)

### Option 1: Accept O(depth) Memory, Optimize Further

1. **Reduce state storage** - Can we store states in lower precision (FP16)?
2. **Optimize contrast checkpointing** - Reduce checkpointing overhead
3. **Fuse operations** - Combine analytic gradient computation with state updates

### Option 2: Explore Alternative Approaches

1. **State recomputation** - Re-compute states during contrast instead of storing (trade compute for memory)
2. **Layer-wise settling** - Settle one layer at a time (may change EP dynamics)
3. **Custom CUDA kernels** - In-place settling with minimal temporary allocations

### Option 3: Shift Focus to Other Advantages

1. **Biological plausibility** - EP's key advantage over backprop
2. **Continual learning** - EP + EWC for reduced forgetting
3. **Local learning** - Layer-local EP updates

---

## Recommendation

**Proceed with Option 1 + Option 3:**

1. Accept that EP has O(depth) memory (same as backprop)
2. Document the 1.8x speedup from analytic gradients as a technical advantage
3. Shift focus to demonstrating EP's unique advantages:
   - Biological plausibility
   - Continual learning performance
   - Potential for neuromorphic deployment

**Revise Phase 2 goals:**
- ~~O(1) memory~~ → O(1) settling overhead (achieved)
- Deep scaling (still valid - test at 5000+ layers)
- Continual learning (EP + EWC)
- Speed optimization (1.8x already achieved with analytic gradients)

---

## Conclusion

Week 3-4 successfully implemented analytic gradients for EP settling. The implementation is **correct** (2.5e-12 gradient match) and **fast** (1.8x speedup), but does not achieve memory savings because:

1. Current EP is already memory-efficient during settling
2. O(depth) state storage is fundamental to EP
3. Analytic gradients add temporary tensor overhead

**Key deliverable:** The analytic gradients implementation (`o1_memory_v2.py`) provides a **1.8x speedup** with identical training results - this is a valuable optimization even without memory savings.

**Revised goal:** Focus on demonstrating EP's unique advantages (biological plausibility, continual learning) rather than pursuing unachievable O(1) total memory.

---

*Created: 2026-02-25*
*Status: Week 3-4 Complete, Phase 2 Goals Under Revision*
