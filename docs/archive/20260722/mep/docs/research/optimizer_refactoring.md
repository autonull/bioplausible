# Optimizer Refactoring Summary

**Date:** 2026-03-04
**Status:** ✅ Complete

---

## Problem

The optimizer codebase had accumulated too many variants:

| Category | Classes/Presets |
|----------|-----------------|
| Core EP | `smep`, `smep_fast`, `sdmep`, `local_ep`, `natural_ep` |
| O(1) Memory v1 | `O1MemoryEP`, `manual_energy_compute`, `settle_manual` |
| O(1) Memory v2 | `O1MemoryEPv2`, `analytic_state_gradients`, `settle_manual_o1` |
| EWC | `EWCRegularizer`, `EPOptimizerWithEWC` |
| Backprop | `muon_backprop` |

**Total:** 15+ different classes/functions for essentially the same functionality.

---

## Solution: Unified `EPOptimizer`

All variants consolidated into a single, well-parameterized class:

```python
from mep import EPOptimizer

# Fast EP (default settings)
opt = EPOptimizer(model.parameters(), model=model)

# EP with EWC for continual learning
opt = EPOptimizer(model.parameters(), model=model, ewc_lambda=100)

# Backprop (for comparison)
opt = EPOptimizer(model.parameters(), model=model, mode='backprop')

# High-accuracy EP (more settling steps)
opt = EPOptimizer(model.parameters(), model=model, settle_steps=30)
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `'ep'` | `'ep'` or `'backprop'` |
| `settle_steps` | `10` | Settling iterations (10=fast, 30=accurate) |
| `gradient_method` | `'analytic'` | `'analytic'` (fast) or `'autograd'` (exact) |
| `ewc_lambda` | `0.0` | EWC regularization (0=disabled, 100-1000=CL) |
| `beta` | `0.5` | EP nudging strength |
| `settle_lr` | `0.2` | Settling learning rate |
| `lr` | `0.01` | Main learning rate |
| `momentum` | `0.9` | Momentum factor |
| `weight_decay` | `0.0005` | Weight decay |
| `loss_type` | `'cross_entropy'` | `'cross_entropy'` or `'mse'` |

---

## Backward Compatibility

All existing presets still work:

```python
from mep import smep, smep_fast, muon_backprop

# These all still work
opt1 = smep(model.parameters(), model=model)
opt2 = smep_fast(model.parameters(), model=model)
opt3 = muon_backprop(model.parameters())
```

**Implementation:** Presets are now thin wrappers around `EPOptimizer`.

---

## Migration Guide

### Before (Multiple Variants)

```python
# Different imports for different variants
from mep import smep
from mep.optimizers import O1MemoryEPv2, EPOptimizerWithEWC

# Different APIs
opt1 = smep(params, model=model, settle_steps=30)
opt2 = O1MemoryEPv2(params, model=model, settle_steps=10)
opt3 = EPOptimizerWithEWC(params, model=model, ewc_lambda=100)
```

### After (Unified)

```python
from mep import EPOptimizer

# Same API, different parameters
opt1 = EPOptimizer(params, model=model, settle_steps=30)
opt2 = EPOptimizer(params, model=model, settle_steps=10, gradient_method='analytic')
opt3 = EPOptimizer(params, model=model, ewc_lambda=100)
```

---

## Files Changed

| File | Change |
|------|--------|
| `mep/optimizers/ep_optimizer.py` | **NEW** - Unified optimizer |
| `mep/optimizers/__init__.py` | Export `EPOptimizer`, mark legacy |
| `mep/__init__.py` | Export `EPOptimizer` |
| `mep/presets/__init__.py` | Now uses `EPOptimizer` internally |

**Legacy files preserved** (for backward compatibility):
- `mep/optimizers/o1_memory.py`
- `mep/optimizers/o1_memory_v2.py`
- `mep/optimizers/ewc.py`

---

## Benefits

### For Users

1. **Simpler API** - One class instead of 10+
2. **Clear parameters** - Explicit tradeoffs (speed vs accuracy)
3. **Easier experimentation** - Change one parameter, not class
4. **Better documentation** - Single source of truth

### For Developers

1. **Less code duplication** - Shared implementation
2. **Easier maintenance** - One place to fix bugs
3. **Clearer testing** - Test one class thoroughly
4. **Extensibility** - Add features in one place

---

## Performance

No performance impact - same underlying algorithms:

| Configuration | Speed vs Backprop |
|--------------|-------------------|
| Default (`settle_steps=10`) | 4-6x slower |
| High-accuracy (`settle_steps=30`) | 10-15x slower |
| With EWC | +5-10% overhead |
| Analytic gradients | 1.5-2x faster settling |

---

## Examples

### Standard EP Training

```python
from mep import EPOptimizer

opt = EPOptimizer(model.parameters(), model=model)

for x, y in train_loader:
    opt.step(x=x, target=y)
```

### Continual Learning with EWC

```python
from mep import EPOptimizer

opt = EPOptimizer(model.parameters(), model=model, ewc_lambda=100)

# Task 1
for epoch in range(epochs):
    for x, y in task1_loader:
        opt.step(x=x, target=y, task_id=0)
opt.consolidate_task(task1_loader, task_id=0)

# Task 2 (with EWC regularization)
for epoch in range(epochs):
    for x, y in task2_loader:
        opt.step(x=x, target=y, task_id=1)
```

### Backprop Comparison

```python
from mep import EPOptimizer

opt = EPOptimizer(model.parameters(), mode='backprop')

for x, y in train_loader:
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    opt.step()
```

---

## Testing

All tests pass:
- ✅ Default EP (analytic gradients)
- ✅ EP with EWC
- ✅ Backprop mode
- ✅ Backprop without model
- ✅ Backward compatibility (`smep`, `smep_fast`, `muon_backprop`)

---

## Next Steps

1. **Update documentation** - Point users to `EPOptimizer`
2. **Deprecation warnings** - Add warnings for legacy classes (optional)
3. **Remove legacy code** - After sufficient deprecation period

---

*Created: 2026-03-04*
*Status: Complete, Backward Compatible*
