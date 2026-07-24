# Optimizer Unification Guide

**Date:** 2026-02-19  
**Status:** ✅ Complete

---

## The Problem: Fragmented Optimizer Landscape

Before unification, optimizers were scattered across multiple modules:

```python
# Confusing: Where do I find optimizers?
from mep import smep  # MEP package
from bioplausible.learning import EqProp  # Learning rules
from torch.optim import Adam  # PyTorch
from bioplausible.hybrid_optimizer import HybridEqPropOptimizer  # Hybrid
```

**Issues:**
- Different namespaces for conceptually the same thing
- Different base classes
- Different calling conventions
- Users must know which module to import from

---

## The Solution: Unified Optimizer Package

All optimizers in one place, inheriting from a common base:

```python
# Clean: All optimizers in one package
from bioplausible.optimizers import (
    FeedbackAlignment,  # Learning rules
    EqProp,  # Learning rules
    smep,  # MEP
    Adam,  # Standard
)

# Unified factory
from bioplausible.optimizers import get_optimizer, create_optimizer

opt = get_optimizer("feedback_alignment", model.parameters(), model=model)
opt = get_optimizer("smep", model.parameters(), model=model)
opt = get_optimizer("adam", model.parameters(), lr=0.001)

# Or simplest:
opt = create_optimizer(model, "feedback_alignment")
opt = create_optimizer(model, "smep")
opt = create_optimizer(model, "adam", lr=0.001)
```

---

## New Structure

```
bioplausible/optimizers/
├── __init__.py              # Unified package exports
│   ├── BioOptimizer         # Common base class
│   ├── get_optimizer()      # Unified factory
│   ├── list_optimizers()    # List all optimizers
│   └── create_optimizer()   # Convenience function
│
└── learning_rules.py        # Learning rule optimizers
    ├── FeedbackAlignment
    ├── DirectFA
    ├── EqProp
    └── ...
```

---

## Optimizer Categories

### 1. Learning Rules (13)

Biologically plausible learning algorithms:

| Optimizer | Description |
|-----------|-------------|
| `feedback_alignment` / `fa` | Fixed random feedback (Lillicrap et al., 2016) |
| `direct_fa` / `dfa` | Direct feedback from output (Nøkland, 2016) |
| `adaptive_fa` | Feedback weights adapt |
| `stochastic_fa` | Noisy feedback weights |
| `contrastive_fa` | Contrastive + FA |
| `eqprop` | Standard EqProp (Scellier & Bengio, 2017) |
| `holomorphic_eqprop` | Complex-valued EqProp |
| `finite_nudge` | Large beta for robustness |
| `lazy_eqprop` | Event-driven updates |
| `chl` / `hebbian` | Contrastive Hebbian Learning |

### 2. MEP Optimizers (6)

Validated MEP optimizers:

| Optimizer | Description |
|-----------|-------------|
| `smep` | Spectral Muon EP (default, validated) |
| `smep_fast` | Fast SMEP (4-6x speedup) |
| `sdmep` | Low-rank SVD for large models |
| `local_ep` | Layer-local learning |
| `natural_ep` | Natural gradient EP |
| `muon_backprop` | Muon + backprop |

### 3. Standard Optimizers (4)

PyTorch standard optimizers:

| Optimizer | Description |
|-----------|-------------|
| `sgd` | SGD with momentum |
| `adam` | Adam |
| `adamw` | AdamW |
| `rmsprop` | RMSprop |

**Total: 23 optimizers** available through a unified interface.

---

## Usage Patterns

### Pattern 1: Direct Import

```python
from bioplausible.optimizers import FeedbackAlignment, EqProp, smep

# Learning rule
opt1 = FeedbackAlignment(model.parameters(), model=model, lr=0.01)

# MEP
opt2 = smep(model.parameters(), model=model, lr=0.01)

# Standard
opt3 = Adam(model.parameters(), lr=0.001)
```

### Pattern 2: Factory Function

```python
from bioplausible.optimizers import get_optimizer

# Any optimizer by name
opt = get_optimizer("feedback_alignment", model.parameters(), model=model)
opt = get_optimizer("smep", model.parameters(), model=model)
opt = get_optimizer("adam", model.parameters(), lr=0.001)
```

### Pattern 3: Convenience Function

```python
from bioplausible.optimizers import create_optimizer

# Simplest: just specify model and optimizer name
opt = create_optimizer(model, "feedback_alignment")
opt = create_optimizer(model, "smep")
opt = create_optimizer(model, "adam", lr=0.001)
```

### Pattern 4: Top-Level Import

```python
# All optimizers also available from bioplausible.*
from bioplausible import FeedbackAlignment, smep, Adam, create_optimizer

opt = create_optimizer(model, "feedback_alignment")
```

---

## Base Class: BioOptimizer

All optimizers inherit from `BioOptimizer`:

```python
from bioplausible.optimizers import BioOptimizer


class MyCustomOptimizer(BioOptimizer):
    def __init__(self, params, model=None, lr=0.01):
        super().__init__(params, model=model, lr=lr)

    def step(self, closure=None, **kwargs):
        # Custom optimization logic
        pass
```

**Key features:**
- Inherits from `torch.optim.Optimizer`
- Compatible with PyTorch ecosystem
- Supports both standard and learning rule patterns

---

## Migration Guide

### Old Code (Deprecated)

```python
# Scattered imports
from mep import smep
from bioplausible.learning import EqProp
from torch.optim import Adam
```

### New Code (Recommended)

```python
# Unified import
from bioplausible.optimizers import smep, EqProp, Adam

# Or use factory
from bioplausible.optimizers import get_optimizer

opt = get_optimizer("smep", model.parameters(), model=model)
```

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Namespaces** | 4+ modules | 1 module |
| **Base classes** | Multiple | 1 (BioOptimizer) |
| **Discovery** | Hard | `list_optimizers()` |
| **Factory** | None | `get_optimizer()` |
| **Convenience** | Manual | `create_optimizer()` |
| **Consistency** | Low | High |

---

## API Reference

### `get_optimizer(name, params, model=None, **kwargs)`

Get any optimizer by name.

```python
opt = get_optimizer("feedback_alignment", model.parameters(), model=model)
```

### `list_optimizers(category=None)`

List available optimizers.

```python
all_opts = list_optimizers()  # All
lr_opts = list_optimizers("learning_rules")  # Learning rules only
mep_opts = list_optimizers("mep")  # MEP only
std_opts = list_optimizers("standard")  # Standard only
```

### `create_optimizer(model, optimizer, **kwargs)`

Create optimizer for a model.

```python
opt = create_optimizer(model, "smep", lr=0.01)
```

---

## Files Changed

| File | Change |
|------|--------|
| `bioplausible/optimizers/__init__.py` | New unified package |
| `bioplausible/optimizers/learning_rules.py` | Moved from `learning.py` |
| `bioplausible/__init__.py` | Export unified optimizers |
| `bioplausible/compat.py` | Updated imports |

---

## Testing

```python
from bioplausible.optimizers import (
    get_optimizer,
    list_optimizers,
    create_optimizer,
)
from bioplausible import ModelZoo

model = ModelZoo.get("looped_mlp", input_dim=784, hidden_dim=256)

# Test all categories
for name in list_optimizers():
    try:
        if name in ["sgd", "adam", "adamw"]:
            opt = get_optimizer(name, model.parameters(), lr=0.01)
        else:
            opt = get_optimizer(name, model.parameters(), model=model)
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
```

---

## Summary

**Before:**
- 4+ modules for optimizers
- Inconsistent APIs
- Confusing for users

**After:**
- 1 module: `bioplausible.optimizers`
- Unified API
- Clear categorization
- Factory functions
- 23 optimizers available

---

*Created: 2026-02-19*  
*Status: Complete and tested*
