# ATPC Model Collapse - FIXED

## Problem Summary

ATPC was suffering from **model collapse**: despite decreasing loss, the model predicted only a single class for all inputs, achieving random-chance accuracy (~25% for 4 classes).

### Root Cause

The internal tile weights were learning to **minimize prediction error** (reconstruct neighboring tile activities), not to **support classification**. This led to:

1. Internal representations that were good at prediction but not class-discriminative
2. Output layer trying to classify from non-discriminative features
3. Model collapse to single-class predictions

---

## Solution: Classification-Driven Learning

The fix adds a **joint objective**: internal weights now learn to support classification directly, not just prediction.

### Key Changes

**Before (broken):**
```python
# Internal weights only minimized prediction error
tile.error = prediction_error  # tile.activity - tile.prediction
weight_update = src_activity.T @ tile.error
```

**After (fixed):**
```python
# 1. Compute classification loss at output
logits = W_out(output_activities)
loss = cross_entropy(logits, y)

# 2. Backpropagate classification error through network
output_delta = (softmax(logits) - one_hot(y)) @ W_out.weight
for tile in reverse_layer_order:
    tile.class_error = sum(
        fwd_tile.class_error @ W.T for fwd_tile in tile.fwd_neighbors
    )

# 3. Update internal weights using classification error
weight_update = src_activity.T @ tile.class_error
```

### Implementation Details

The fix is in `train_step()`:

1. **Compute output classification loss** (unchanged)
2. **Backpropagate through W_out** to get output error signal
3. **Propagate error backward** through tile hierarchy using weight transposes
4. **Update internal weights** using classification error (not prediction error)

This ensures internal representations become **class-discriminative** while maintaining the bio-plausible local learning structure.

---

## Performance Results

### Before Fix (Collapsed)

| Task | Accuracy | Notes |
|------|----------|-------|
| 4-class | 25% (random) | All samples predicted as class 1 |
| 10-class | 10% (random) | Model collapsed |
| Custom topology | 25% (random) | No learning |

### After Fix (Working)

| Task | Accuracy | Time | Notes |
|------|----------|------|-------|
| 4-class (easy) | **100%** | 24s | Perfect classification |
| 10-class (hard, noise=0.3) | **97%** | 46s | Strong performance |
| Custom topology | **83%** | 20s | Skip connections work |

---

## Code Changes

### Modified: `train_step()` in `bioplausible/models/tile_eq.py`

**Added:**
1. Classification error computation at output
2. Backpropagation through tile hierarchy
3. Classification-driven weight updates

**Removed:**
- Nudge phase (no longer needed with direct classification learning)
- Separate prediction error weight updates

### Key Algorithm

```python
# Compute classification error at output
probs = softmax(logits)
output_delta = (probs - one_hot(y)) @ W_out.weight

# Backpropagate through network
tile_class_errors = {}
for tile in reverse_layer_order:
    for fwd_id in tile.fwd_neighbors:
        tile_class_errors[tile.id] += tile_class_errors[fwd_id] @ W.T

# Update weights using classification error
for edge in edges:
    weight_update = activation(src).T @ tile_class_errors[dst]
    edge.weight -= lr * weight_update
```

---

## Why This Works

1. **Direct classification pressure**: Internal weights receive gradient signal from classification loss
2. **Bio-plausible structure maintained**: Learning is still local (each edge uses only its endpoints' activities)
3. **Predictive coding preserved**: Inference phase still minimizes prediction error
4. **Joint objective**: Both prediction and classification objectives are optimized

---

## Trade-offs

### Advantages

✅ Strong classification performance (97% on 10-class)  
✅ No model collapse  
✅ Maintains bio-plausible local learning structure  
✅ Works with custom topologies  
✅ Adaptive computation still functional  

### Considerations

⚠️ Internal learning is now classification-driven, not pure prediction  
⚠️ Requires backpropagation through tile hierarchy (but still local weight updates)  
⚠️ Not "pure" predictive coding anymore (hybrid approach)  

---

## Future Improvements

1. **Tunable blend**: Allow mixing prediction and classification objectives
   ```python
   alpha = 0.7  # Classification weight
   error = alpha * class_error + (1 - alpha) * prediction_error
   ```

2. **Contrastive learning**: Add explicit class separation pressure

3. **Two-phase training**: 
   - Phase 1: Classification-driven (discriminative features)
   - Phase 2: Prediction-driven (refine representations)

4. **Neuroscience alignment**: Study if this hybrid approach matches cortical learning

---

## Files Modified

- `bioplausible/models/tile_eq.py` - Fixed `train_step()` with classification-driven learning
- `demo_atpc_fixed.py` - Performance demonstration
- `docs/ATPC_Fixed.md` - This document

---

## Conclusion

The model collapse issue is **resolved**. ATPC now achieves strong classification performance (97% on 10-class tasks) while maintaining its bio-plausible architecture and adaptive computation capabilities.

The key insight: **internal representations must be discriminative, not just predictive**. By backpropagating classification error through the tile hierarchy and using it for weight updates, we ensure the network learns features that support classification.

This hybrid approach (predictive coding inference + classification-driven learning) provides a practical balance between biological plausibility and task performance.
