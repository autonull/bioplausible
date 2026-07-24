# MEP Integration Guide

**Date:** 2026-02-19  
**Status:** Integrated into Bioplausible

---

## Executive Summary

MEP (Muon Equilibrium Propagation) has been successfully integrated into the Bioplausible framework. This integration combines:

- **MEP's validated optimizer strategies** (91-94% MNIST in 3 epochs)
- **Bioplausible's EqProp models** (30+ algorithm variants)
- **Unified Zoo registry** for models and optimizers
- **Hybrid optimizer** leveraging both codebases

---

## Quick Start

### Using the Zoo (Recommended)

```python
from bioplausible import ModelZoo, OptimizerZoo, SupervisedTrainer

# Get a model from the zoo
model = ModelZoo.get("looped_mlp", input_size=784, hidden_size=256, output_size=10)

# Get an optimizer from the zoo
optimizer = OptimizerZoo.get("smep", model.parameters(), model=model)

# Train
trainer = SupervisedTrainer(model, device="cuda")
trainer.fit(train_loader, val_loader, epochs=10)
```

### Direct MEP Import

```python
from bioplausible import smep, LoopedMLP

model = LoopedMLP(784, 256, 10)
optimizer = smep(model.parameters(), model=model, mode="ep")

for x, y in train_loader:
    optimizer.step(x=x, target=y)
```

### Hybrid Optimizer

```python
from bioplausible import HybridEqPropOptimizer, LoopedMLP

model = LoopedMLP(784, 256, 10)

optimizer = HybridEqPropOptimizer(
    model.parameters(),
    model=model,
    lr=0.01,
    settle_steps=30,
    use_triton=True,  # Use Triton backend if available
)

for x, y in train_loader:
    optimizer.step(x=x, target=y)
```

---

## Available Models

List all models:
```python
from bioplausible import list_models

print(list_models())  # All models
print(list_models("eqprop"))  # EqProp variants
print(list_models("feedback_alignment"))  # FA family
```

### EqProp Models (Core)

| Model | Description | Tags |
|-------|-------------|------|
| `looped_mlp` | Standard looped MLP with spectral norm | vision, lm, stable |
| `conv_eqprop` | Convolutional EqProp for vision | vision, cnn |
| `transformer_eqprop` | Transformer with EqProp dynamics | lm, attention |
| `memory_efficient_mlp` | Gradient checkpointing for deep nets | deep, memory_efficient |
| `modern_conv_eqprop` | Residual ConvEqProp (CIFAR-10 optimized) | vision, sota |

### Advanced EqProp Variants

| Model | Description | Tags |
|-------|-------------|------|
| `holomorphic_eqprop` | Complex-valued EqProp (NeurIPS 2024) | complex, research |
| `finite_nudge_eqprop` | Large beta for noise robustness | robust |
| `lazy_eqprop` | Event-driven updates (97% FLOP reduction) | efficient |
| `sparse_eqprop` | Top-K sparsity during settling | sparse, biological |
| `momentum_eqprop` | Momentum for faster settling | fast |

### Feedback Alignment Family

| Model | Description | Tags |
|-------|-------------|------|
| `feedback_alignment` | Fixed random feedback weights | bio_plausible |
| `direct_fa` | Direct feedback from output | skip_connection |
| `stochastic_fa` | Noise in feedback weights | robust, stochastic |
| `adaptive_fa` | Feedback weights adapt | adaptive |
| `contrastive_fa` | Contrastive + FA | contrastive |

### Hebbian & Hybrid

| Model | Description | Tags |
|-------|-------------|------|
| `hebbian_chain` | Deep Hebbian chain (500+ layers) | deep, local_learning |
| `contrastive_hebbian` | CHL (precursor to EqProp) | contrastive |
| `predictive_coding_hybrid` | EqProp + Predictive Coding | hybrid |
| `eqprop_diffusion` | Energy-based diffusion | generative |

---

## Available Optimizers

List all optimizers:
```python
from bioplausible import list_optimizers

print(list_optimizers())  # All optimizers
print(list_optimizers("ep"))  # EP variants
print(list_optimizers("backprop"))  # Backprop
```

### MEP Optimizers (Validated)

| Optimizer | Category | Description | Speed |
|-----------|----------|-------------|-------|
| `smep` | ep | Spectral Muon EP (default) | 10-15x slower than BP |
| `smep_fast` | ep | Fast SMEP (4-6x speedup) | 4-6x slower than BP |
| `sdmep` | ep | Low-rank SVD for large models | Varies |
| `local_ep` | ep | Layer-local learning | 10-15x slower than BP |
| `natural_ep` | natural_gradient | Fisher whitening | 15-20x slower than BP |
| `muon_backprop` | backprop | Muon + backprop | 1.2x slower than BP |

### Standard Optimizers

| Optimizer | Category | Description |
|-----------|----------|-------------|
| `sgd` | backprop | SGD with momentum |
| `adam` | backprop | Adam (default baseline) |
| `adamw` | backprop | AdamW with decoupled WD |

---

## Optimizer Selection Guide

### For Standard Training (Recommended)

```python
# Best overall: validated, stable
optimizer = OptimizerZoo.get("smep", model.parameters(), model=model)
```

### For Fast Training

```python
# 4-6x speedup with minimal accuracy loss
optimizer = OptimizerZoo.get("smep_fast", model.parameters(), model=model)
```

### For Large Models (>100M params)

```python
# Low-rank SVD reduces memory
optimizer = OptimizerZoo.get("sdmep", model.parameters(), model=model)
```

### For Biological Plausibility

```python
# Each layer uses only local information
optimizer = OptimizerZoo.get("local_ep", model.parameters(), model=model)
```

### For Baseline Comparison

```python
# Standard backprop with Muon orthogonalization
optimizer = OptimizerZoo.get("muon_backprop", model.parameters(), model=model)
```

---

## Configuration Guide

### SMEP Parameters

```python
optimizer = OptimizerZoo.get(
    "smep",
    model.parameters(),
    model=model,
    lr=0.01,  # Learning rate
    momentum=0.9,  # Momentum factor
    weight_decay=0.0005,  # Weight decay
    mode="ep",  # 'ep' or 'backprop'
    settle_steps=30,  # Settling iterations (reduce to 10 for speed)
    settle_lr=0.15,  # Settling learning rate
    beta=0.5,  # Nudging strength (0.3-0.7)
    loss_type="mse",  # 'mse' or 'cross_entropy'
    ns_steps=5,  # Newton-Schulz iterations
    gamma=0.95,  # Spectral norm bound
)
```

### Performance Tuning

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `settle_steps` | More steps = better settling, slower | 10-50 |
| `settle_lr` | Higher = faster convergence | 0.1-0.2 |
| `beta` | Nudging strength | 0.3-0.7 |
| `ns_steps` | Muon orthogonalization quality | 3-7 |
| `gamma` | Spectral norm bound | 0.9-0.99 |

---

## Performance Expectations

### MNIST Classification

| Epochs | Target Accuracy | Expected with SMEP |
|--------|-----------------|--------------------|
| 1 | >80% | 90-92% |
| 3 | >88% | 91-94% |
| 10 | >90% | 95-96% |

### Speed Comparison

| Optimizer | Relative Speed | Best For |
|-----------|----------------|----------|
| Backprop (Adam) | 1.0x | Baseline |
| Muon Backprop | 1.2x slower | Drop-in SGD replacement |
| SMEP-Fast | 4-6x slower | Fast EP training |
| SMEP | 10-15x slower | Best accuracy |
| Natural EP | 15-20x slower | Research |

---

## Testing

### Smoke Test (20 seconds)

```bash
python -m pytest tests/test_mep_smoke.py -v
```

### Full Regression (3 minutes)

```bash
python -m pytest tests/test_mep_regression.py -v
```

### Performance Benchmark

```bash
python -m pytest tests/test_mep_performance.py -v
```

---

## Migration from Standalone MEP

If you have existing code using standalone MEP:

### Before (Standalone MEP)

```python
from mep import smep

model = MyModel()
optimizer = smep(model.parameters(), model=model)
```

### After (Bioplausible Integration)

```python
from bioplausible import smep  # Same import!

model = MyModel()
optimizer = smep(model.parameters(), model=model)
```

**Backward compatibility is maintained!** All existing MEP code should work without changes.

---

## Advanced: Custom Strategy Composition

For advanced users, you can compose custom optimizers:

```python
from bioplausible import (
    CompositeOptimizer,
    EPGradient,
    MuonUpdate,
    SpectralConstraint,
    ErrorFeedback,
)

optimizer = CompositeOptimizer(
    model.parameters(),
    gradient=EPGradient(beta=0.5, settle_steps=30),
    update=MuonUpdate(ns_steps=5),
    constraint=SpectralConstraint(gamma=0.95),
    feedback=ErrorFeedback(beta=0.9),  # Optional
    lr=0.01,
    model=model,
)
```

---

## Troubleshooting

### "MEP import failed"

Ensure MEP is installed:
```bash
pip install -e mep/
```

### "Model not found in Zoo"

Check available models:
```python
from bioplausible import list_models

print(list_models())
```

### Slow Training

Try `smep_fast` for 4-6x speedup:
```python
optimizer = OptimizerZoo.get("smep_fast", model.parameters(), model=model)
```

### NaN/Inf in Gradients

Reduce learning rate or settling steps:
```python
optimizer = OptimizerZoo.get(
    "smep",
    model.parameters(),
    model=model,
    settle_lr=0.1,  # Reduced from 0.15
    settle_steps=20,  # Reduced from 30
)
```

---

## What Was Integrated

### Core Components (✅ Validated)

- `CompositeOptimizer` - Strategy pattern optimizer
- `EPGradient` - EP gradient computation
- `MuonUpdate` - Newton-Schulz orthogonalization
- `SpectralConstraint` - Spectral norm constraints
- `Settler` - Settling dynamics
- `EnergyFunction` - EP energy computation
- `ModelInspector` - Model structure extraction

### Presets (✅ Validated)

- `smep` - Default SMEP
- `smep_fast` - Fast SMEP
- `sdmep` - Low-rank SDMEP
- `local_ep` - Local EP
- `natural_ep` - Natural gradient EP
- `muon_backprop` - Muon + backprop

### NOT Integrated (Experimental)

- `EPOptimizer` (unified) - Broken (52-76% accuracy)
- `O1MemoryEP` - O(1) memory not achieved
- `O1MemoryEPv2` - Untested
- `EWCRegularizer` - Needs validation

---

## Future Work

### Planned Enhancements

1. **Adaptive settling** - Early stopping when converged
2. **Custom CUDA kernels** - Fused settling operations
3. **Better weight initialization** - EP-specific init strategies
4. **Continual learning** - EP + EWC integration

### Research Directions

1. **Holomorphic EqProp** - Complex-valued states for exact gradients
2. **Directed EP** - Asymmetric forward/backward weights
3. **Quantized EP** - INT8/ternary weights for edge devices

---

## References

- [MEP Repository](../mep/README.md)
- [Bioplausible README](../README.md)
- [Scientist Guide](../SCIENTIST_GUIDE.md)
- [PHASE2_FINAL_SUMMARY](../mep/PHASE2_FINAL_SUMMARY.md)

---

*Created: 2026-02-19*  
*Status: Integrated and validated*
