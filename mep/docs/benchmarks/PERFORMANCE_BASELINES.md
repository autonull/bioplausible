# MEP Performance Baselines

This document establishes performance baselines for the MEP optimizer to prevent regression during future development.

## Baseline Results (Validated 2026-02-18)

### Classification Benchmarks

#### MNIST - mlp_small (No Dropout)
| Epochs | Optimizer | Test Accuracy | Time/Epoch | Notes |
|--------|-----------|---------------|------------|-------|
| 1 | SMEP | 90-93% | 4-5s | Quick sanity check |
| 3 | SMEP | 91-94% | 4-5s | Standard benchmark |
| 3 | SGD | 90-93% | 2-3s | Backprop baseline |
| 3 | Adam | 89-92% | 2-3s | Backprop baseline |
| 10 | SMEP | 95-96% | 4-5s | Extended training |
| 10 | Adam | 95-96% | 2-3s | Extended training |

**Minimum Threshold (Regression Test):**
- 1 epoch: ≥80%
- 3 epochs: ≥88%
- 10 epochs: ≥90%

#### MNIST - MLP with Dropout
| Epochs | Optimizer | Test Accuracy | Notes |
|--------|-----------|---------------|-------|
| 3 | SMEP | 85-90% | Dropout slows EP settling |
| 3 | SGD | 90-93% | Dropout works well with BP |

**Note:** Dropout is incompatible with EP settling. Use models without dropout for EP.

#### XOR Problem
| Steps | Optimizer | Accuracy | Notes |
|-------|-----------|----------|-------|
| 50 | SMEP | ≥90% | Basic convergence test |
| 100 | SMEP | ≥95% | Full convergence |
| 200 | SMEP | 100% | Perfect classification |

**Minimum Threshold:** 100 steps, ≥75% accuracy

### Continual Learning Benchmarks

#### Permuted MNIST (2 tasks, simple)
| Method | Task 1 Acc | Task 2 Acc | Forgetting | Notes |
|--------|------------|------------|------------|-------|
| EP (no EF) | 90-95% | 90-95% | 40-50% | Standard catastrophic forgetting |
| EP + EF | 80-85% | 85-90% | 30-40% | EF reduces forgetting but slows learning |
| EWC | 90-95% | 90-95% | 5-15% | Best overall |

**Key Finding:** Error feedback reduces forgetting but also reduces initial learning speed. EWC is more effective for continual learning.

### Memory Benchmarks

#### Activation Memory vs Depth (with Gradient Checkpointing)
| Depth | Backprop (MB) | EP (MB) | EP Overhead |
|-------|--------------|---------|-------------|
| 100 | ~20 | ~32 | +60% |
| 500 | ~26 | ~85 | +220% |
| 1000 | ~35 | ~150 | +330% |
| 2000 | ~51 | ~285 | +460% |

**Key Finding:** EP uses MORE memory than backprop+checkpointing, not less. The O(1) memory claim is refuted.

### Speed Benchmarks

#### Relative Training Speed (MNIST, 1 epoch)
| Optimizer | Relative Time | Notes |
|-----------|---------------|-------|
| SGD | 1.0× | Baseline |
| Adam | 1.0× | Similar to SGD |
| SMEP (default, 30 steps) | 10-15× | Default settling settings |
| SMEP (optimized, 10 steps) | 4-6× | Reduced settling steps |
| SMEP (analytic gradients) | 3-5× | With o1_memory_v2 |
| EQPROP | 1.5-2.0× | No Muon, fewer settling steps |

**Key Finding:** EP speed is **proportional to settling steps**. The 2-3× figure assumes optimized settings:
- 10-15 settling steps (vs 30 default)
- Analytic gradients (o1_memory_v2.py)
- Adaptive settling (early stopping)

**Default settings (30 steps, autograd gradients) result in 10-15× slower training.**

#### Speed Optimization Guide

```python
# Default (13x slower)
optimizer = smep(model.parameters(), model=model, settle_steps=30)

# Optimized (4-6x slower)
optimizer = smep(model.parameters(), model=model, settle_steps=10, settle_lr=0.2)

# With analytic gradients (3-5x slower)
from mep.optimizers import O1MemoryEPv2
optimizer = O1MemoryEPv2(model.parameters(), model=model, settle_steps=10)
```

## Optimal Configuration

For classification tasks, the following configuration achieves best results:

```python
from mep import smep

optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    # Critical settling parameters
    beta=0.5,           # Nudging strength (0.3-0.7 range)
    settle_steps=30,    # Settling iterations (20-50 range)
    settle_lr=0.15,     # Settling LR (0.1-0.2 range)
    # Stability settings
    loss_type='mse',    # More stable than cross_entropy
    use_error_feedback=False,  # Disable for classification
    # Regularization
    gamma=0.95,         # Spectral norm bound
    ns_steps=5,         # Muon orthogonalization steps
)
```

## Model Architecture Guidelines

### Compatible with EP
- ✅ Linear/Dense layers
- ✅ Conv2d layers
- ✅ ReLU, Sigmoid, Tanh activations
- ✅ LayerNorm, BatchNorm
- ✅ Skip connections / residuals
- ✅ MultiheadAttention

### Incompatible with EP
- ❌ Dropout (breaks settling convergence)
- ❌ Stochastic layers during settling

### Recommended Architecture
```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    # No Dropout!
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

## Regression Testing

Run performance regression tests before merging changes:

```bash
# Quick tests (~30 seconds)
pytest tests/regression/test_performance_baseline.py -v

# Full benchmark suite (~5 minutes)
python -m mep.benchmarks.tuned_compare --epochs 3 --model mlp_small
```

### Pass Criteria
- All performance tests pass
- MNIST 3-epoch accuracy ≥88%
- XOR 100-step accuracy ≥75%
- No new test failures

## Known Limitations

1. **Memory**: EP uses more memory than backprop+checkpointing
2. **Speed**: EP is 2-3× slower due to settling
3. **Dropout**: Incompatible with EP settling
4. **Continual Learning**: Error feedback helps but EWC is more effective

## References

- `tests/regression/test_performance_baseline.py` - Automated regression tests
- `VALIDATION_RESULTS.md` - Full validation study
- `ROADMAP.md` - Future research directions
