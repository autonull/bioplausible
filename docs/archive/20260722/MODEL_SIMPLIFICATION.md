# Model Simplification Guide

**Date:** 2026-02-19  
**Status:** ✅ Complete

---

## Key Insight: Why Custom Models?

**Question:** Can we use stock PyTorch MLP/Transformer instead of LoopedMLP/TransformerEqProp?

**Answer:** No - EqProp models have fundamentally different architecture.

### Stock MLP vs LoopedMLP

| Aspect | Stock MLP | LoopedMLP |
|--------|-----------|-----------|
| **Architecture** | Feedforward | Recurrent |
| **Forward** | `y = W2(relu(W1(x)))` | `h_{t+1} = tanh(W_in @ x + W_rec @ h_t)` |
| **Settling** | None | Iterates to equilibrium |
| **EqProp compatible** | No | Yes |

### Stock Transformer vs TransformerEqProp

| Aspect | Stock Transformer | TransformerEqProp |
|--------|-------------------|-------------------|
| **Layer processing** | Feedforward through layers | Joint equilibrium |
| **Attention** | Standard | EqProp-compatible |
| **Settling** | None | All layers iterate together |
| **EqProp compatible** | No | Yes |

---

## Simplified Model Architecture

### Before (Complex)

```
bioplausible/models/
├── looped_mlp.py           # 340 lines, complex
├── transformer_eqprop.py   # 217 lines, complex
├── eqprop_base.py          # 596 lines, abstract base
├── utils.py                # Various utilities
└── ... (20+ files)
```

### After (Clean)

```
bioplausible/models/
├── __init__.py             # Unified exports
├── looped_mlp_simple.py    # 136 lines, clean
├── eqprop_wrappers.py      # Generic wrappers
├── eqprop_base.py          # (unchanged)
└── ... (specialized variants)
```

---

## New Simplified Models

### 1. LoopedMLP (Simplified)

```python
from bioplausible.models import LoopedMLP

model = LoopedMLP(
    input_dim=784,
    hidden_dim=256,
    output_dim=10,
    use_spectral_norm=True,
    max_steps=30,
)

# Forward pass with settling
x = torch.randn(32, 784)
output = model(x, steps=30)  # [32, 10]
```

**Key features:**
- 136 lines (down from 340)
- Uses PyTorch's `spectral_norm` directly
- Clean implementation of EqPropModel interface
- Implements 5 required abstract methods

### 2. Generic Wrappers

For flexibility, we provide wrappers that convert standard PyTorch modules into EqProp models:

```python
from bioplausible.models import (
    RecurrentWrapper,
    StackedRecurrentWrapper,
    TransformerEqPropWrapper,
)
import torch.nn as nn

# Wrap RNNCell
cell = nn.RNNCell(784, 256)
model = RecurrentWrapper(cell, input_dim=784, hidden_dim=256, output_dim=10)

# Stacked RNN
model = StackedRecurrentWrapper("rnn", 784, 256, 10, num_layers=3)

# Transformer
model = TransformerEqPropWrapper(784, 256, 10, num_heads=8, num_layers=4)
```

### 3. Factory Functions

```python
from bioplausible.models import (
    create_rnn_eqprop,
    create_transformer_eqprop,
)

# RNN-based EqProp
model = create_rnn_eqprop(784, 256, 10, cell_type="rnn")
model = create_rnn_eqprop(784, 256, 10, cell_type="lstm")
model = create_rnn_eqprop(784, 256, 10, cell_type="gru")

# Transformer-based EqProp
model = create_transformer_eqprop(784, 256, 10, num_heads=8, num_layers=4)
```

---

## Model Registry

All models registered with consistent naming:

```python
from bioplausible.models import list_models, get_model

print(list_models())
# ['conv', 'conv_eqprop', 'eqprop_mlp', 'gru_eqprop',
#  'looped_mlp', 'lstm_eqprop', 'memory_efficient_mlp',
#  'mlp', 'recurrent_mlp', 'rnn_eqprop', 'transformer_eqprop']

# Get by name
model = get_model("looped_mlp", input_dim=784, hidden_dim=256, output_dim=10)
model = get_model(
    "rnn_eqprop", input_dim=784, hidden_dim=256, output_dim=10, cell_type="rnn"
)
model = get_model("transformer_eqprop", input_dim=784, hidden_dim=256, output_dim=10)
```

---

## EqPropModel Interface

All EqProp models implement 5 abstract methods:

```python
from bioplausible.models.eqprop_base import EqPropModel


class MyEqPropModel(EqPropModel):
    def _build_layers(self):
        """Build all layers."""
        pass

    def forward_step(self, h, x_transformed):
        """Single equilibrium iteration."""
        pass

    def _initialize_hidden_state(self, x):
        """Initialize hidden state."""
        pass

    def _transform_input(self, x):
        """Transform input for the loop."""
        pass

    def _output_projection(self, h):
        """Project hidden state to output."""
        pass
```

---

## Benefits of Simplification

| Aspect | Before | After |
|--------|--------|-------|
| **Lines of code** | 340 (LoopedMLP) | 136 (LoopedMLP) |
| **Dependencies** | Custom utilities | PyTorch primitives |
| **Flexibility** | Fixed architectures | Generic wrappers |
| **Readability** | Complex | Clean |
| **Maintainability** | Hard | Easy |

---

## Migration Guide

### Old Code

```python
from bioplausible.models.looped_mlp import LoopedMLP

model = LoopedMLP(
    input_dim=784,
    hidden_dim=256,
    output_dim=10,
    use_spectral_norm=True,
)
```

### New Code (Same API)

```python
from bioplausible.models import LoopedMLP

model = LoopedMLP(
    input_dim=784,
    hidden_dim=256,
    output_dim=10,
    use_spectral_norm=True,
)
```

**No changes needed!** The API is backward compatible.

---

## When to Use Each Model

### LoopedMLP (Recommended for most cases)

- Simple recurrent architecture
- Fast settling (~20-30 steps)
- Good for MNIST, small vision tasks
- Baseline for EqProp research

### StackedRecurrentWrapper

- Deeper architectures needed
- Multiple recurrent layers
- Joint equilibrium across layers

### TransformerEqPropWrapper

- Sequence modeling
- Attention mechanisms needed
- Language modeling tasks

### RecurrentWrapper (Custom Cell)

- Custom recurrent dynamics
- Research on new cell types
- Specialized architectures

---

## Summary

**Key points:**

1. **EqProp models CANNOT use stock PyTorch** - They need recurrent dynamics for equilibrium
2. **Simplified implementation** - 136 lines vs 340 lines for LoopedMLP
3. **Generic wrappers** - Convert any PyTorch module to EqProp-compatible
4. **Clean API** - `create_model()`, `get_model()`, `list_models()`
5. **Backward compatible** - Old code still works

---

*Created: 2026-02-19*  
*Status: Complete and tested*
