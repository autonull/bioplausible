# Bioplausible API Stability Guide

**Date:** 2026-02-19  
**Status:** ✅ Stable for External Dependencies and Scientist

---

## Overview

The Bioplausible API has been stabilized for use by:
- External dependencies (packages that import bioplausible)
- Scientist processes (AutoScientist autonomous experiments)
- Validation tracks (automated verification)

---

## Stable API Surfaces

### 1. Model API (`bioplausible.models`)

**Factory Functions (Recommended):**
```python
from bioplausible.models import create_model, get_model, list_models

# Create any model
model = create_model('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
model = get_model('conv_eqprop', input_channels=3, output_dim=10)

# List available
models = list_models()  # 20 models available
```

**Direct Class Imports:**
```python
from bioplausible.models import (
    # Core EqProp
    LoopedMLP,
    ConvEqProp,
    MemoryEfficientLoopedMLP,
    
    # Legacy (backward compatible)
    BackpropMLP,
    TransformerEqProp,
    NeuralCube,
    DeepHebbianChain,
    # ... 20+ models total
)

model = LoopedMLP(input_dim=784, hidden_dim=256, output_dim=10)
```

**Available Models (20):**
- Core: `looped_mlp`, `mlp`, `eqprop_mlp`, `recurrent_mlp`
- Wrappers: `rnn_eqprop`, `lstm_eqprop`, `gru_eqprop`, `transformer_eqprop`
- Legacy: `backprop_mlp`, `conv_eqprop`, `memory_efficient_mlp`, `transformer`, `neural_cube`, `hebbian_chain`, `chl`, `lazy_eqprop`, `finite_nudge`, `holomorphic`, `directed_ep`, `feedback_alignment`, `adaptive_fa`, `direct_fa`, `stochastic_fa`, `contrastive_fa`

---

### 2. Optimizer API (`bioplausible.optimizers`)

**Factory Functions (Recommended):**
```python
from bioplausible.optimizers import create_optimizer, get_optimizer, list_optimizers

# Create any optimizer
opt = create_optimizer(model, 'smep')
opt = create_optimizer(model, 'feedback_alignment')
opt = create_optimizer(model, 'adam', lr=0.001)

# List available
optims = list_optimizers()  # 23 optimizers available
```

**Direct Class Imports:**
```python
from bioplausible.optimizers import (
    # Learning rules
    FeedbackAlignment,
    DirectFA,
    EqProp,
    HolomorphicEqProp,
    FiniteNudgeEqProp,
    LazyEqProp,
    ContrastiveHebbianLearning,
    
    # MEP
    smep,
    smep_fast,
    sdmep,
    local_ep,
    natural_ep,
    muon_backprop,
    
    # Standard
    SGD,
    Adam,
    AdamW,
)

opt = FeedbackAlignment(model.parameters(), model=model)
```

**Available Optimizers (23):**
- Learning rules (13): `feedback_alignment`, `fa`, `direct_fa`, `dfa`, `adaptive_fa`, `stochastic_fa`, `contrastive_fa`, `eqprop`, `holomorphic_eqprop`, `finite_nudge`, `lazy_eqprop`, `chl`, `hebbian`
- MEP (6): `smep`, `smep_fast`, `sdmep`, `local_ep`, `natural_ep`, `muon_backprop`
- Standard (4): `sgd`, `adam`, `adamw`, `rmsprop`

---

### 3. Top-Level API (`bioplausible.*`)

**Most Common (Recommended for External Users):**
```python
from bioplausible import (
    # Simplest API
    create_model,
    create_optimizer,
    list_models,
    list_optimizers,
    
    # Training
    SupervisedTrainer,
    EqPropTrainer,
    
    # Data
    get_vision_dataset,
    get_lm_dataset,
)
```

**All Top-Level Exports (~50):**
- Models: `LoopedMLP`, `ConvEqProp`, `MemoryEfficientLoopedMLP`
- Optimizers: `FeedbackAlignment`, `EqProp`, `smep`, `smep_fast`, `SGD`, `Adam`, `AdamW`
- Training: `SupervisedTrainer`, `EqPropTrainer`
- Data: `get_vision_dataset`, `get_lm_dataset`, `create_data_loaders`
- Utilities: `count_parameters`, `verify_spectral_norm`, `TRITON_AVAILABLE`
- Advanced: `ExperimentRunner`, `HyperparameterSearch`, `TrainingVisualizer`, `ResultAnalyzer`, `ModelExporter`, `InferenceEngine`

---

## Scientist Compatibility

The AutoScientist can use all stable APIs:

```python
# Scientist experiment configuration
from bioplausible import (
    create_model,
    create_optimizer,
    SupervisedTrainer,
    get_vision_dataset,
)

# Model search space
model_configs = [
    {'name': 'looped_mlp', 'input_dim': 784, 'hidden_dim': 64, 'output_dim': 10},
    {'name': 'looped_mlp', 'input_dim': 784, 'hidden_dim': 128, 'output_dim': 10},
    {'name': 'looped_mlp', 'input_dim': 784, 'hidden_dim': 256, 'output_dim': 10},
]

# Optimizer search space
optimizer_configs = [
    {'name': 'smep', 'lr': 0.01, 'beta': 0.5},
    {'name': 'feedback_alignment', 'lr': 0.01},
    {'name': 'adam', 'lr': 0.001},
]

# Run experiment
for model_cfg in model_configs:
    model = create_model(**model_cfg)
    
    for opt_cfg in optimizer_configs:
        opt = create_optimizer(model, opt_cfg['name'], **{k: v for k, v in opt_cfg.items() if k != 'name'})
        
        trainer = SupervisedTrainer(model, device='cuda')
        trainer.fit(train_loader, val_loader, epochs=10)
```

---

## External Dependency Compatibility

External packages can rely on stable imports:

```python
# my_package/integration.py
import bioplausible

# All these are stable
model = bioplausible.create_model('looped_mlp', ...)
opt = bioplausible.create_optimizer(model, 'smep')

# Or direct imports
from bioplausible import LoopedMLP, smep, SupervisedTrainer
```

**Version Compatibility:**
- Bioplausible 0.2.0+ provides stable API
- Backward compatibility maintained for legacy imports
- Deprecation warnings shown for old patterns

---

## Validation Track Compatibility

All validation tracks can import models:

```python
from bioplausible.models import (
    # Core
    LoopedMLP,
    ConvEqProp,
    MemoryEfficientLoopedMLP,
    
    # Legacy (all backward compatible)
    BackpropMLP,
    TransformerEqProp,
    NeuralCube,
    DeepHebbianChain,
    ContrastiveHebbianLearning,
    LazyEqProp,
    FiniteNudgeEP,
    HolomorphicEP,
    DirectedEP,
    FeedbackAlignmentEqProp,
    AdaptiveFeedbackAlignment,
    DirectFeedbackAlignmentEqProp,
    StochasticFA,
    ContrastiveFeedbackAlignment,
    BackpropTransformerLM,
    CausalTransformerEqProp,
    EqPropDiffusion,
    ModernConvEqProp,
    # Aliases
    AdaptiveFA,
    EqPropAttentionOnlyLM,
    EquilibriumAlignment,
    FullEqPropLM,
    HybridEqPropLM,
    LoopedMLPForLM,
    RecurrentEqPropLM,
)
```

---

## API Consistency Patterns

### Naming Convention

| Pattern | Functions |
|---------|-----------|
| `create_*` | `create_model()`, `create_optimizer()` |
| `get_*` | `get_model()`, `get_optimizer()`, `get_vision_dataset()` |
| `list_*` | `list_models()`, `list_optimizers()`, `list_presets()` |

### Return Types

| Function | Returns |
|----------|---------|
| `create_model()` | `nn.Module` instance |
| `get_model()` | `nn.Module` instance |
| `list_models()` | `List[str]` |
| `create_optimizer()` | `Optimizer` instance |
| `get_optimizer()` | `Optimizer` instance |
| `list_optimizers()` | `List[str]` |

### Error Handling

```python
from bioplausible.models import create_model

try:
    model = create_model('unknown_model', ...)
except ValueError as e:
    # Clear error message with available options
    print(e)  # "Unknown model: unknown_model\nAvailable: looped_mlp, ..."
```

---

## Testing API Stability

```python
def test_api_stability():
    """Test that core APIs are stable."""
    from bioplausible import (
        create_model, create_optimizer,
        list_models, list_optimizers,
    )
    
    # Models
    assert len(list_models()) > 0
    model = create_model('looped_mlp', input_dim=10, hidden_dim=20, output_dim=5)
    assert model is not None
    
    # Optimizers
    assert len(list_optimizers()) > 0
    opt = create_optimizer(model, 'smep')
    assert opt is not None
    
    print("✓ API stability verified")
```

---

## Migration from Old API

### Old (Deprecated)
```python
from bioplausible.zoo import ModelZoo, OptimizerZoo

model = ModelZoo.get('looped_mlp', input_dim=784, hidden_dim=256)
opt = OptimizerZoo.get('smep', model.parameters(), model=model)
```

### New (Recommended)
```python
from bioplausible import create_model, create_optimizer

model = create_model('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
opt = create_optimizer(model, 'smep')
```

**Note:** Old API still works but new API is simpler.

---

## Summary

| Aspect | Status |
|--------|--------|
| **Model API** | ✅ Stable (20 models) |
| **Optimizer API** | ✅ Stable (23 optimizers) |
| **Top-level API** | ✅ Stable (~50 exports) |
| **Scientist Compatible** | ✅ Yes |
| **External Dependencies** | ✅ Yes |
| **Validation Tracks** | ✅ Yes (backward compatible) |
| **Naming Consistency** | ✅ `create_*`, `get_*`, `list_*` |
| **Error Messages** | ✅ Clear with options |

---

*Created: 2026-02-19*  
*Status: Stable for production use*
