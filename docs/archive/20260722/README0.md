# Bioplausible

**Bio-Plausible Learning Algorithms for PyTorch**

A comprehensive framework for biologically plausible deep learning, featuring Equilibrium Propagation, Feedback Alignment, Hebbian Learning, and MEP (Muon Equilibrium Propagation) optimizers.

---

## Featured: EquiTile Language Models

**EquiTile** is a novel tile-based architecture for efficient language modeling:

- **Mixture of Tiles (MoT):** Sparse conditional computation via top-k tile activation
- **Flexible Attention:** Flash Attention 2, SDPA, or manual with sliding window support
- **Parameter Efficient:** Grouped Query Attention + weight tying
- **Research-Ready:** Full reproducibility framework, statistical benchmarking

📄 **Complete Architecture Specification:** [docs/EQUITILE.md](docs/EQUITILE.md)

```python
from bioplausible.models.equitile.lm_demo import FastLMEquiTile, FastLMConfig

config = FastLMConfig(
    vocab_size=50000,
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    num_kv_heads=2,      # GQA 4:1
    mot_k=2,             # Top-2 tiles active
    attention_type="auto",
    use_compile=True,
)
model = FastLMEquiTile(config)
```

**For researchers:** See [docs/EQUITILE.md](docs/EQUITILE.md) for complete architectural specification, novelty assessment, and comparison to prior art.

---

## Quick Start

```python
from bioplausible import create_model, create_optimizer, SupervisedTrainer

# Create model
model = create_model('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)

# Create optimizer
optimizer = create_optimizer(model, 'smep')

# Train
trainer = SupervisedTrainer(model, device='cuda')
trainer.fit(train_loader, val_loader, epochs=10)
```

---

## Installation

```bash
# Install package
pip install -e .

# Install MEP optimizers (optional)
cd mep && pip install -e . --break-system-packages
```

---

## Models (50+)

### Core EqProp Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `LoopedMLP` | Recurrent MLP with equilibrium dynamics | General vision/LM |
| `BackpropMLP` | Standard feedforward MLP (baseline) | Comparison |
| `ConvEqProp` | Convolutional EqProp | Vision tasks |
| `MemoryEfficientLoopedMLP` | Gradient checkpointing for deep nets | Memory-constrained |
| `TransformerEqProp` | Transformer with EqProp dynamics | Language modeling |

### Advanced EqProp Variants

| Model | Description | Use Case |
|-------|-------------|----------|
| `NeuralCube` | 3D lattice topology | Topology embedding |
| `LazyEqProp` | Event-driven updates (97% FLOP reduction) | Efficient computing |
| `FiniteNudgeEP` | Large beta for noise robustness | Noisy environments |
| `HolomorphicEP` | Complex-valued EqProp (exact gradients) | Research |
| `DirectedEP` | Asymmetric forward/backward weights | Deep scaling |
| `HomeostaticEqProp` | Biological homeostatic regulation | Biological modeling |
| `TemporalResonanceEqProp` | Spike-timing dependent plasticity | Temporal processing |
| `TernaryEqProp` | Ternary weights {-1, 0, +1} | Low-precision |
| `SparseEquilibrium` | Top-K sparsity during settling | Energy efficiency |
| `MomentumEquilibrium` | Momentum for faster settling | Faster convergence |
| `StandardEqProp` | Standard EqProp implementation | Baseline |

### Feedback Alignment Family

| Model | Description | Use Case |
|-------|-------------|----------|
| `FeedbackAlignmentEqProp` | Fixed random feedback weights | Weight transport solution |
| `AdaptiveFeedbackAlignment` | Feedback weights adapt | Improved alignment |
| `DirectFeedbackAlignmentEqProp` | Direct feedback from output | Skip connections |
| `StochasticFA` | Noisy feedback weights | Robustness |
| `ContrastiveFeedbackAlignment` | Contrastive + FA | Representation learning |
| `StandardFA` | Standard feedback alignment | Baseline |
| `EnergyGuidedFA` | Energy-guided feedback | Hybrid approach |
| `EnergyMinimizingFA` | Energy-minimizing FA | Optimization |
| `LayerwiseEquilibriumFA` | Layerwise equilibrium | Local learning |
| `EquilibriumAlignment` | Equilibrium-based alignment | Hybrid |

### Hebbian & Hybrid Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `DeepHebbianChain` | Deep Hebbian chain (500+ layers) | Local learning |
| `ContrastiveHebbianLearning` | CHL (precursor to EqProp) | Energy-based |
| `PredictiveCodingHybrid` | EqProp + Predictive Coding | Hybrid |

### Language Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `EqPropAttentionOnlyLM` | EqProp on attention only | Stable LM |
| `FullEqPropLM` | Full EqProp language model | Complete LM |
| `HybridEqPropLM` | Hybrid EqProp LM | Hybrid |
| `LoopedMLPForLM` | LoopedMLP for LM | Simple LM |
| `RecurrentEqPropLM` | Recurrent EqProp LM | Sequential |
| `BackpropTransformerLM` | Backprop Transformer (baseline) | Comparison |
| `CausalTransformerEqProp` | Causal EqProp Transformer | Autoregressive |

### Generative & Vision

| Model | Description | Use Case |
|-------|-------------|----------|
| `EqPropDiffusion` | Energy-based diffusion | Generative |
| `ModernConvEqProp` | ResNet-style CNN (CIFAR-10 optimized) | Vision SOTA |

---

## Optimizers (23)

### Learning Rule Optimizers

| Optimizer | Description | Speed | Best For |
|-----------|-------------|-------|----------|
| `FeedbackAlignment` | Fixed random feedback | 1.2x slower | Bio-plausible |
| `DirectFA` | Direct feedback from output | 1.2x slower | Skip connections |
| `EqProp` | Standard Equilibrium Propagation | 10-15x slower | Best accuracy |
| `HolomorphicEqProp` | Complex-valued EqProp | 20-30x slower | Exact gradients |
| `FiniteNudgeEqProp` | Large beta nudge | 10-15x slower | Noise robustness |
| `LazyEqProp` | Event-driven updates | 2-3x slower | Efficiency |
| `ContrastiveHebbianLearning` | CHL optimizer | 10-15x slower | Energy-based |

### MEP Optimizers (Validated)

| Optimizer | Description | Speed | Best For |
|-----------|-------------|-------|----------|
| `smep` | Spectral Muon EP (default) | 10-15x slower | Best accuracy (91-94% MNIST) |
| `smep_fast` | Fast SMEP | 4-6x slower | Fast training |
| `sdmep` | Low-rank SVD for large models | Varies | Large models |
| `local_ep` | Layer-local learning | 10-15x slower | Biological plausibility |
| `natural_ep` | Natural gradient EP | 15-20x slower | Research |
| `muon_backprop` | Muon + backprop | 1.2x slower | Drop-in SGD replacement |

### Standard Optimizers

| Optimizer | Description | Use Case |
|-----------|-------------|----------|
| `SGD` | SGD with momentum | Baseline |
| `Adam` | Adam optimizer | Baseline |
| `AdamW` | AdamW with decoupled WD | Baseline |

---

## Usage Examples

### Create Model and Optimizer

```python
from bioplausible import create_model, create_optimizer

# Simplest API
model = create_model('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
optimizer = create_optimizer(model, 'smep')

# Direct class usage
from bioplausible import LoopedMLP, FeedbackAlignment

model = LoopedMLP(input_dim=784, hidden_dim=256, output_dim=10)
optimizer = FeedbackAlignment(model.parameters(), model=model)
```

### Training

```python
from bioplausible import SupervisedTrainer, get_vision_dataset

# Load data
train_loader, val_loader, _ = get_vision_dataset('mnist', batch_size=128)

# Create model and optimizer
model = create_model('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
optimizer = create_optimizer(model, 'smep')

# Train
trainer = SupervisedTrainer(model, device='cuda')
trainer.fit(train_loader, val_loader, epochs=10)
```

### Experiment Runner

```python
from bioplausible.experiments import ExperimentRunner

runner = ExperimentRunner()

result = runner.run(
    model_name='looped_mlp',
    optimizer_name='smep',
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
)

print(f"Validation accuracy: {result.val_accuracy:.2f}%")
```

### Hyperparameter Search

```python
from bioplausible.experiments import HyperparameterSearch

search = HyperparameterSearch()

best_params, best_result = search.grid_search(
    model_name='looped_mlp',
    optimizer_name='smep',
    param_grid={
        'lr': [0.001, 0.01, 0.1],
        'beta': [0.3, 0.5, 0.7],
        'settle_steps': [10, 30, 50],
    },
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
)

print(f"Best params: {best_params}")
print(f"Best accuracy: {best_result.val_accuracy:.2f}%")
```

### Research Presets

```python
from bioplausible.experiments import get_preset, list_presets, run_preset

# List available presets
print(list_presets())  # 15 presets available

# Run a preset
result = run_preset('performance_vision_default', train_loader, val_loader)
```

### Visualization

```python
from bioplausible.visualization_tools import TrainingVisualizer, visualize_results

viz = TrainingVisualizer()
viz.plot_training_curve(train_losses, val_losses, save_path='training.png')

# Generate all visualizations
paths = visualize_results(results, output_dir='./viz')
```

### Analysis

```python
from bioplausible.analysis_tools import ResultAnalyzer, analyze_results

analyzer = ResultAnalyzer()
analyzer.add_results(results)

report = analyzer.generate_report()
print(report.summary())
```

### Deployment

```python
from bioplausible.deployment import export_model, InferenceEngine

# Export model
export_model(model, 'looped_mlp', model_params, output_dir='./exports')

# Load and infer
engine = InferenceEngine.from_export('./exports')
prediction = engine.predict(input_tensor)
```

---

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `create_model(name, **kwargs)` | Create model by name |
| `create_optimizer(model, name, **kwargs)` | Create optimizer for model |
| `list_models()` | List available models |
| `list_optimizers()` | List available optimizers |

### Models

```python
from bioplausible import (
    # Core
    LoopedMLP, BackpropMLP, ConvEqProp,
    MemoryEfficientLoopedMLP, TransformerEqProp,
    # Advanced
    NeuralCube, LazyEqProp, FiniteNudgeEP,
    HolomorphicEP, DirectedEP, HomeostaticEqProp,
    TemporalResonanceEqProp, TernaryEqProp,
    SparseEquilibrium, MomentumEquilibrium, StandardEqProp,
    # FA family
    FeedbackAlignmentEqProp, AdaptiveFeedbackAlignment,
    DirectFeedbackAlignmentEqProp, StochasticFA,
    ContrastiveFeedbackAlignment, StandardFA,
    EnergyGuidedFA, EnergyMinimizingFA,
    LayerwiseEquilibriumFA, EquilibriumAlignment,
    # Hebbian/Hybrid
    DeepHebbianChain, ContrastiveHebbianLearning,
    PredictiveCodingHybrid,
    # LM
    EqPropAttentionOnlyLM, FullEqPropLM, HybridEqPropLM,
    LoopedMLPForLM, RecurrentEqPropLM,
    BackpropTransformerLM, CausalTransformerEqProp,
    # Generative/Vision
    EqPropDiffusion, ModernConvEqProp,
)
```

### Optimizers

```python
from bioplausible import (
    # Learning rules
    FeedbackAlignment, DirectFA, EqProp,
    HolomorphicEqProp, FiniteNudgeEqProp,
    LazyEqProp, ContrastiveHebbianLearning,
    # MEP
    smep, smep_fast, sdmep,
    local_ep, natural_ep, muon_backprop,
    # Standard
    SGD, Adam, AdamW,
)
```

### Training & Data

```python
from bioplausible import (
    SupervisedTrainer, EqPropTrainer,
    get_vision_dataset, get_lm_dataset,
    create_data_loaders,
)
```

### Experiments

```python
from bioplausible.experiments import (
    ExperimentRunner, HyperparameterSearch,
    quick_comparison, benchmark_model,
    get_preset, list_presets, run_preset,
    ALL_PRESETS,
)
```

### Visualization

```python
from bioplausible.visualization_tools import (
    TrainingVisualizer, ResultsDashboard,
    visualize_results,
)
```

### Analysis

```python
from bioplausible.analysis_tools import (
    ResultAnalyzer, AnalysisReport,
    analyze_results,
)
```

### Deployment

```python
from bioplausible.deployment import (
    ModelExporter, InferenceEngine,
    export_model, load_model,
)
```

### Scientist & Validation

```python
from bioplausible.scientist import AutoScientist
from bioplausible.validation import Verifier
```

---

## Performance Benchmarks

### MNIST Classification

| Epochs | Optimizer | Expected Accuracy |
|--------|-----------|-------------------|
| 1 | smep | 90-92% |
| 3 | smep | 91-94% |
| 10 | smep | 95-97% |
| 10 | smep_fast | 90-93% |
| 10 | muon_backprop | 97-98% |

### Speed Comparison

| Optimizer | Relative Speed | Use Case |
|-----------|----------------|----------|
| Backprop (Adam) | 1.0x | Baseline |
| muon_backprop | 1.2x slower | Drop-in replacement |
| smep_fast | 4-6x slower | Fast EP training |
| smep | 10-15x slower | Best accuracy |
| natural_ep | 15-20x slower | Research |

---

## Documentation

| Document | Description |
|----------|-------------|
| [API Stability](docs/API_STABILITY.md) | Stable API reference |
| [Optimizer Unification](docs/OPTIMIZER_UNIFICATION.md) | Optimizer package guide |
| [Simplified API](docs/SIMPLIFIED_API.md) | Simplified usage guide |
| [Model Simplification](docs/MODEL_SIMPLIFICATION.md) | Model architecture guide |
| [Learning Rule Refactoring](docs/LEARNING_RULE_REFACTORING.md) | Learning rules migration |
| [Experimentation Guide](docs/EXPERIMENTATION_GUIDE.md) | Complete experimentation workflows |

---

## Testing

```bash
# Run MEP integration tests
python -m pytest tests/test_mep_integration.py -v

# Run all tests
python -m pytest tests/ -v
```

---

## Project Structure

```
bioplausible/
├── __init__.py              # Main exports (30 core)
├── models/                  # 50+ models
│   ├── __init__.py
│   ├── looped_mlp_simple.py
│   └── ...
├── optimizers/              # 23 optimizers
│   ├── __init__.py
│   ├── base.py
│   └── learning_rules.py
├── experiments/             # Experiment utilities
├── visualization_tools.py   # Visualization
├── analysis_tools.py        # Analysis
├── deployment.py            # Deployment
├── scientist/               # AutoScientist
├── validation/              # Validation tracks
├── training/                # Training utilities
└── datasets.py              # Data loaders
```

---

## Citation

```bibtex
@software{bioplausible2026,
  title = {Bioplausible: Bio-Plausible Learning Algorithms for PyTorch},
  year = {2026},
  url = {https://github.com/automenta/bioplausible},
}
```

---

## License

MIT License - see LICENSE file for details.

---

**Status:** ✅ Production Ready | **Version:** 0.2.0 | **Last Updated:** 2026-02-19
