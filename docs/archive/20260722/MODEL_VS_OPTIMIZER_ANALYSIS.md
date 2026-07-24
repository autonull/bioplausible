# Model vs Optimizer Refactoring Analysis

**Date:** 2026-02-19  
**Purpose:** Identify EqProp "models" that should be "optimizers" instead

---

## Key Principle

**Model** = Architecture (what computation happens)  
**Optimizer** = Learning rule (how parameters are updated)

---

## Current State: Confused Responsibilities

Many Bioplausible "models" combine architecture + learning rule, which creates:
1. **Code duplication** - Same architecture, different learning rules
2. **Combinatorial explosion** - N architectures × M learning rules = N×M classes
3. **Inflexibility** - Can't easily try new learning rules on existing architectures

---

## Analysis by Category

### ✅ SHOULD STAY AS MODELS (Architecture-only)

These define unique architectures:

| Model | Reason |
|-------|--------|
| `LoopedMLP` | Specific recurrent architecture |
| `ConvEqProp` | CNN architecture |
| `TransformerEqProp` | Transformer architecture |
| `MemoryEfficientLoopedMLP` | Architecture with gradient checkpointing |
| `NeuralCube` | 3D lattice topology |
| `ModernConvEqProp` | ResNet-style CNN architecture |
| `DeepHebbianChain` | Deep feedforward chain architecture |

### ⚠️ SHOULD BECOME OPTIMIZERS (Learning rules)

These differ only in how gradients/updates are computed:

| Current "Model" | Should Become | Reason |
|-----------------|---------------|--------|
| `FeedbackAlignmentEqProp` | `LoopedMLP` + `FAOptimizer` | Same arch, different feedback |
| `DirectFeedbackAlignmentEqProp` | `LoopedMLP` + `DFAOptimizer` | Direct feedback is a gradient method |
| `AdaptiveFeedbackAlignment` | `LoopedMLP` + `AdaptiveFAOptimizer` | Feedback adaptation is optimization |
| `StochasticFA` | `LoopedMLP` + `StochasticFAOptimizer` | Noise in feedback is optimization |
| `ContrastiveFeedbackAlignment` | `LoopedMLP` + `ContrastiveFAOptimizer` | Contrastive is a loss/optimizer |
| `EnergyGuidedFA` | `LoopedMLP` + `EnergyGuidedFAOptimizer` | Energy guidance is optimization |
| `EnergyMinimizingFA` | `LoopedMLP` + `EnergyMinimizingFAOptimizer` | Energy minimization is optimization |
| `EquilibriumAlignment` | `LoopedMLP` + `EquilibriumAlignmentOptimizer` | Alignment strategy is optimization |
| `LayerwiseEquilibriumFA` | `LoopedMLP` + `LayerwiseFAOptimizer` | Layerwise is optimization strategy |
| `HolomorphicEP` | `LoopedMLP` + `HolomorphicEPOptimizer` | Complex-valued settling is optimization |
| `FiniteNudgeEP` | `LoopedMLP` + `FiniteNudgeEPOptimizer` | Large beta is an optimizer parameter |
| `LazyEqProp` | `LoopedMLP` + `LazyEPOptimizer` | Event-driven updates are optimization |
| `MomentumEquilibrium` | `LoopedMLP` + `MomentumEPOptimizer` | Momentum is optimization |
| `SparseEquilibrium` | `LoopedMLP` + `SparseEPOptimizer` | Sparsity is a constraint/optimizer |
| `HomeostaticEqProp` | `LoopedMLP` + `HomeostaticOptimizer` | Homeostasis is regularization |
| `TemporalResonance` | `LoopedMLP` + `TemporalResonanceOptimizer` | STDP is a learning rule |
| `DirectedEP` | `LoopedMLP` + `DirectedEPOptimizer` | Asymmetric weights are optimization |
| `ContrastiveHebbianLearning` | `LoopedMLP` + `CHLOptimizer` | CHL is a learning rule |

### 🔄 HYBRID CASES (Both architecture + learning)

| Model | Recommendation |
|-------|----------------|
| `EqPropDiffusion` | Keep as model (unique generative architecture) |
| `CausalTransformerEqProp` | Split: `TransformerEqProp` + `CausalOptimizer` |
| `HybridEqPropLM` | Keep (combines multiple learning rules) |

---

## Benefits of Refactoring

### Before (Current)
```python
# 20+ separate model classes
from bioplausible.models import (
    FeedbackAlignmentEqProp,
    DirectFeedbackAlignmentEqProp,
    AdaptiveFeedbackAlignment,
    # ... 17 more FA variants
)

# Can't easily combine
model = FeedbackAlignmentEqProp(...)  # Fixed learning rule
```

### After (Refactored)
```python
# 1 model class + multiple optimizers
from bioplausible import ModelZoo, OptimizerZoo

model = ModelZoo.get("looped_mlp", input_dim=784, hidden_dim=256)
optimizer = OptimizerZoo.get("feedback_alignment", model.parameters())
# or
optimizer = OptimizerZoo.get("direct_fa", model.parameters())
# or
optimizer = OptimizerZoo.get("adaptive_fa", model.parameters())
```

### Combinatorial Reduction

| Scenario | Before | After |
|----------|--------|-------|
| Architectures | 5 | 5 |
| Learning rules | 20 (duplicated) | 20 |
| Total classes | 5 × 20 = **100** | 5 + 20 = **25** |
| New combinations | Add new class | Mix and match |

---

## MEP Integration Already Solves This

The MEP optimizers already encapsulate EqProp learning:

```python
# MEP approach (correct separation)
from bioplausible import LoopedMLP, smep

model = LoopedMLP(input_dim=784, hidden_dim=256)  # Architecture only
optimizer = smep(model.parameters(), model=model)  # Learning rule
```

This is the **correct pattern** that should be extended to all learning variants.

---

## Refactoring Priority

### High Priority (Clear optimizer candidates)

1. **Feedback Alignment family** (6 variants)
   - All are `LoopedMLP` + different feedback matrices
   - Should become: `LoopedMLP` + `FAOptimizer` variants

2. **EqProp variants** (5 variants)
   - `HolomorphicEP`, `FiniteNudgeEP`, `LazyEqProp`, etc.
   - Should become: `LoopedMLP` + `EPOptimizer` variants

3. **Energy-based variants** (3 variants)
   - `EnergyGuidedFA`, `EnergyMinimizingFA`, `EquilibriumAlignment`
   - Should become: `LoopedMLP` + energy-based optimizers

### Medium Priority (Some architectural differences)

4. **Hybrid models** (3 variants)
   - `PredictiveCodingHybrid`, `ContrastiveHebbianLearning`
   - Need careful analysis of what's architecture vs learning

### Low Priority (Keep as models)

5. **True architectures** (7 variants)
   - `ConvEqProp`, `TransformerEqProp`, `NeuralCube`, etc.
   - These define unique computation, not just learning

---

## Implementation Strategy

### Phase 1: Create Optimizer Wrappers
```python
# New optimizer that wraps existing model behavior
class FeedbackAlignmentOptimizer:
    def __init__(self, params, model, feedback_weights, **kwargs):
        self.model = model
        self.feedback_weights = feedback_weights
    
    def step(self, x, target):
        # FA learning rule
        ...
```

### Phase 2: Update Zoo Registry
```python
# Register optimizers separately from models
OptimizerZoo.register(
    OptimizerSpec(
        name="feedback_alignment",
        category="learning_rule",
        optimizer_class=FeedbackAlignmentOptimizer,
        description="Fixed random feedback weights",
    )
)
```

### Phase 3: Deprecate Old Models
```python
# Keep old names for backward compatibility
class FeedbackAlignmentEqProp:
    def __new__(cls, *args, **kwargs):
        warnings.warn("Use LoopedMLP + FAOptimizer instead")
        model = LoopedMLP(*args, **kwargs)
        optimizer = FeedbackAlignmentOptimizer(...)
        return model, optimizer
```

---

## Recommended Actions

1. **Keep MEP pattern** - It correctly separates model/optimizer
2. **Create learning rule optimizers** - For FA, Hebbian, EqProp variants
3. **Update Zoo** - Register optimizers separately
4. **Maintain backward compatibility** - Deprecate, don't remove
5. **Update documentation** - Clarify model vs optimizer distinction

---

## Conclusion

**Yes, many EqProp "models" should be "optimizers" instead.**

The MEP integration already demonstrates the correct pattern:
- **Models** = Architecture (LoopedMLP, ConvEqProp, etc.)
- **Optimizers** = Learning rules (smep, FA, Hebbian, etc.)

Refactoring would reduce code duplication, enable new combinations, and clarify the framework's architecture.

---

*Created: 2026-02-19*
