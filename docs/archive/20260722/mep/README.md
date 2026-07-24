# MEP: Muon Equilibrium Propagation

### 🧠 Biologically Plausible Deep Learning Without Backpropagation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/your-username/mep/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/mep)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Vision

**Equilibrium Propagation (EP)** offers a fundamentally different approach to neural network training: instead of backpropagating errors through a computation graph, networks settle to energy minima, and gradients emerge from the contrast between equilibrium states.

**MEP** enhances EP with geometry-aware optimization—combining EP's biological plausibility with modern techniques for stable, efficient training. Our goal: make biologically plausible learning practical for real-world applications while opening new research directions in neuromorphic computing, continual learning, and energy-efficient AI.

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
import torch.nn as nn
from mep import smep, muon_backprop

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# Option 1: EP mode (biologically plausible)
optimizer = smep(model.parameters(), model=model, mode="ep")
optimizer.step(x=x, target=y)  # No .backward() needed!

# Option 2: Backprop mode (drop-in replacement)
optimizer = muon_backprop(model.parameters())
loss.backward()
optimizer.step()
```

### Optimal EP Configuration

```python
from mep import smep

optimizer = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode="ep",
    beta=0.5,  # Nudging strength
    settle_steps=30,  # Settling iterations
    settle_lr=0.15,  # Settling learning rate
    loss_type="mse",  # Stable energy
    use_error_feedback=False,
)
```

---

## How MEP Works

### Equilibrium Propagation Foundation

EP trains networks through an energy-based formulation:

1. **Free Phase (β=0):** Input is presented, network settles to equilibrium by minimizing internal energy
2. **Nudged Phase (β>0):** Target gently nudges the output, network settles to a new equilibrium
3. **Gradient from Contrast:** The difference between free and nudged states approximates the gradient

```
Free Phase:                    Nudged Phase:
Input → [Layers] → Output      Input → [Layers] → Output ← Target (nudge)
        ↓ settles                       ↓ settles
      states*                         states^β
        
Gradient = (states^β - states*) / β
```

**Key advantage:** No backward pass through the computation graph. Learning uses only local information at each layer.

### The MEP Enhancement: S-D-M

MEP adds three key innovations to stabilize and accelerate EP:

#### **S — Spectral Constraints**

EP requires contractive dynamics for stable settling. We enforce this through spectral normalization:

```python
σ(W) ≤ γ < 1  # Spectral radius bounded
```

This guarantees convergence to a unique fixed point, eliminating the oscillatory behavior that plagued early EP implementations.

**Implementation:** Power iteration after each update, enforcing σ(W) ≤ 0.95 by default.

#### **D — Dion Low-Rank Updates**

For large weight matrices (>100K parameters), we use low-rank SVD approximation:

```python
G ≈ U @ S @ V^T
update = U @ V^T  # Scale-invariant orthogonal update
```

This reduces computational cost while preserving gradient information in the dominant subspace. Error feedback accumulates residuals to recover lost information over time.

**Benefit:** Enables EP to scale to larger models without prohibitive compute costs.

#### **M — Muon Orthogonalization**

We apply Newton-Schulz iteration to orthogonalize gradients before applying updates:

```python
X_{k+1} = ½ X_k (3I - X_k^T X_k)
```

This improves conditioning and enables stable training at greater depths, similar to how batch normalization helps backprop but without the additional parameters.

**Benefit:** Better gradient flow, faster convergence, more stable training.

### The Complete MEP Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  MEP Training Step                                          │
├─────────────────────────────────────────────────────────────┤
│  1. Free Phase: Settle network with β=0                     │
│     - Iterative energy minimization                         │
│     - States converge to fixed point s*                     │
│                                                             │
│  2. Nudged Phase: Settle network with β>0                   │
│     - Target nudges output layer                            │
│     - States converge to s^β                                │
│                                                             │
│  3. Contrast: Compute gradient                              │
│     - ∇L ≈ (s^β - s*) / β                                   │
│     - Gradients flow through contrast, not backprop         │
│                                                             │
│  4. Transform Gradient:                                     │
│     - Dion: Low-rank SVD for large matrices                 │
│     - Muon: Newton-Schulz orthogonalization                 │
│     - Error feedback: accumulate residuals                  │
│                                                             │
│  5. Apply Update:                                           │
│     - Momentum buffer                                       │
│     - Spectral constraint enforcement                       │
│     - Weight decay                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance

| Benchmark | EP | SGD | Adam |
|-----------|-----|-----|------|
| MNIST (3 epoch) | **91.4%** | 91.0% | 90.2% |
| MNIST (10 epoch) | 95.37% | 93.80% | **95.75%** |
| XOR (100 step) | 100% | 100% | 100% |

📊 **Full results:** [docs/benchmarks/VALIDATION_RESULTS.md](docs/benchmarks/VALIDATION_RESULTS.md)

---

## Why MEP Matters

### Biological Plausibility

Backpropagation has a fundamental problem: it requires symmetric forward and backward weights (the "weight transport problem"). This is biologically implausible—real brains don't have access to exact transpose weights.

**EP solves this:** Learning uses only local information. Each layer updates based on its own activity contrast, not global error signals.

**Potential impact:** Better models of biological learning, insights into how real brains learn.

### Neuromorphic Computing

Digital backpropagation is energy-inefficient on emerging analog hardware. EP's local learning rules and event-based dynamics map naturally to:

- **Optical chips** — continuous-time dynamics
- **Memristor arrays** — local weight updates
- **SpiNNaker/Loihi** — asynchronous event-based processing

**Potential impact:** Energy-efficient AI on specialized hardware.

### Continual Learning

EP's energy-based formulation and error feedback mechanisms may offer advantages for learning sequential tasks without catastrophic forgetting.

**Potential impact:** Agents that learn continuously like humans do.

### Research Tool

MEP provides a well-tested, performant implementation for researchers studying:
- Alternative learning mechanisms
- Energy-based models
- Local learning rules
- Non-backprop architectures

---

## When to Use MEP

### ✅ Ideal For:
- Biological plausibility research
- Neuromorphic hardware deployment
- Energy-based model research
- Continual learning experiments
- Educational demonstrations
- Exploring alternatives to backprop

### ✅ Also Good For:
- Standard classification (performance matches backprop)
- Research prototypes
- Novel architectures

### ⚠️ Consider Backprop For:
- Production deployment (mature tooling)
- Speed-critical training
- Very large-scale models (until MEP scales further)

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/index.md](docs/index.md) | **Start here** — Full documentation index |
| [docs/benchmarks/PERFORMANCE_BASELINES.md](docs/benchmarks/PERFORMANCE_BASELINES.md) | Performance thresholds, optimal config |
| [docs/benchmarks/VALIDATION_RESULTS.md](docs/benchmarks/VALIDATION_RESULTS.md) | Full validation study |
| [docs/research/ROADMAP_RESEARCH.md](docs/research/ROADMAP_RESEARCH.md) | Research trajectory, partnerships |
| [docs/methods_paper.md](docs/methods_paper.md) | Preprint-ready methods paper |

---

## Examples

| Example | Description |
|---------|-------------|
| `examples/quickstart.py` | Minimal working example |
| `examples/demo_ep_vs_backprop.py` | EP vs backprop comparison |
| `examples/mnist_comparison.py` | MNIST classification demo |
| `examples/train_char_lm.py` | Character-level LM training |

---

## Research Roadmap

### Phase 1: Foundation ✅ (Q1 2026)
- [x] Performance parity achieved (~95% MNIST)
- [x] 156 tests passing
- [x] Optimal parameters discovered
- [x] Documentation complete

### Phase 2: Technical Excellence (Q2-Q3 2026) - IN PROGRESS
- [ ] **O(1) memory implementation** - Avoid PyTorch activation overhead
- [ ] **Deep network scaling** - Train 5000-10000+ layer networks
- [ ] **Continual learning** - EP+EWC integration
- [ ] **Speed optimization** - Reduce settling overhead

### Phase 3: Results & Outreach (Q4 2026+)
- [ ] Neuromorphic hardware demos
- [ ] Biological plausibility studies
- [ ] Community building

📋 **Full roadmap:** [docs/research/ROADMAP_RESEARCH.md](docs/research/ROADMAP_RESEARCH.md)

---

## Contributing

Contributions welcome! High-priority areas:

1. **Neuromorphic demos** — Run MEP on Loihi, SpiNNaker, or similar
2. **Continual learning** — EP + EWC integration
3. **Architecture exploration** — What works best with EP?
4. **Energy profiling** — Quantify efficiency vs backprop
5. **Documentation** — Tutorials, examples, guides

See [docs/research/ROADMAP_RESEARCH.md](docs/research/ROADMAP_RESEARCH.md) for collaboration opportunities.

---

## Citation

```bibtex
@software{mep2026,
  title = {MEP: Muon Equilibrium Propagation},
  author = {MEP Contributors},
  year = {2026},
  url = {https://github.com/your-username/mep},
}
```

---

## Acknowledgments

- Equilibrium Propagation: Scellier & Bengio (2017)
- Muon Optimizer: Keller Jordan (2024)
- Spectral Normalization: Miyato et al. (2018)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Last updated: 2026-02-18*
