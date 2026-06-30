# Bioplausible

Bioplausible - where neuroscience and deep learning converge. This is a comprehensive framework for exploring locally-learning, energy-efficient neural networks through the power of PyTorch, PyTorch Lightning, and Optuna. Bioplausible transcends the limits of conventional backpropagation, offering a rich ecosystem of plausibly biological learning algorithms, novel architectures like the `EquiTile` system, and a suite of tools for autonomous scientific discovery. It is designed to be an inspiring and powerful ally for researchers and practitioners alike.

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Theoretical Foundations](#theoretical-foundations)
- [Architecture & Algorithms](#architecture--algorithms)
- [Optimization & Training](#optimization--training)
- [Distributed Training & P2P](#distributed-training--p2p)
- [Deployment & Inference](#deployment--inference)
- [Validation & Testing](#validation--testing)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

## Features

- 🌿 **Bio-Plausible Learning**: Explore a rich zoo of learning algorithms beyond conventional backpropagation, including Equilibrium Propagation (EqProp), Feedback Alignment, and Hebbian Learning.
- 🧩 **EquiTile Architecture**: Experience our novel tile-based architecture for efficient, local computation. This phonotype is adapted for Vision, Language, Reinforcement Learning, Graphs, and Time-Series.
- 🚀 **PyTorch-First**: Built from the ground up with PyTorch for maximum flexibility and performance, with seamless integration with the PyTorch Lightning ecosystem.
- 🧭 **Lightning-Integration**: Leverage PyTorch Lightning for structured and scalable training workflows, including automatic hardware management, logging, and distributed training support.
- 🎯 **Optuna Hyperparameter Search**: Find optimal configurations efficiently with integrated Optuna-powered search, complete with advanced samplers and pruners.
- ⚙️ **Strategy-Pattern Optimizers**: Our MEP (Muon Equilibrium Propagation) optimizers are built on a composable strategy pattern, allowing for flexible customization of gradient processing, updates, and feedback.
- 🧠 **AutoScientist Agent**: An autonomous experimentalist that manages resources, selects strategies, and synthesizes results to guide the research process.
- 🔬 **Comprehensive Validation**: Validate your models with a suite of specialized tracks, from quick smoke tests to deep-dive research benchmarks.
- 🌐 **Distributed & P2P Ready**: Train your models across multiple machines with a flexible P2P coordinator system or via the Lightning Trainer.
- 💻 **PyQt6 Desktop GUI**: An intuitive desktop application for managing experiments, visualizing results, and interacting with the framework.
- 🛠️ **Production-Ready Utilities**: Tools for model export (ONNX, TorchScript), inference, and deployment.

## Quick Start

First, make sure you have installed the package as described below. Then, run the following code in a Python environment:

```python
from bioplausible import create_model, create_optimizer, SupervisedTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Prepare Data ---
# Replace with your actual data loading logic
inputs = torch.randn(256, 784)
labels = torch.randint(0, 10, (256,))
train_dataset = TensorDataset(inputs, labels)
train_loader = DataLoader(train_dataset, batch_size=32)

val_inputs = torch.randn(64, 784)
val_labels = torch.randint(0, 10, (64,))
val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=32)

# --- 2. Create Model ---
# We start with a simple LoopedMLP model.
model = create_model('looped_mlp', input_dim=784, hidden_dim=128, output_dim=10)

# --- 3. Create Optimizer ---
# We use the 'smep' (Spectral Muon Equilibrium Propagation) optimizer.
optimizer = create_optimizer(model, 'smep')

# --- 4. Train with SupervisedTrainer ---
trainer = SupervisedTrainer(model=model, optimizer=optimizer, device='cuda')
trainer.fit(train_loader, val_loader, epochs=10)
```

### Running a Hyperparameter Search with Optuna

Want to find the best hyperparameters automatically? Bioplausible integrates with Optuna to make this a breeze.

```python
from bioplausible.experiments import HyperparameterSearch
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Prepare Data (using dummy data for the example) ---
inputs = torch.randn(256, 784)
labels = torch.randint(0, 10, (256,))
train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=32)
val_inputs = torch.randn(64, 784)
val_labels = torch.randint(0, 10, (64,))
val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=32)

# --- 2. Run Search ---
search = HyperparameterSearch()
# Define the grid of hyperparameters to search over
best_params, best_result = search.grid_search(
    model_name='looped_mlp',
    optimizer_name='smep',
    param_grid={
        'lr': [0.001, 0.01, 0.1],
        'beta': [0.5, 1.0, 2.0],
    },
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
)

print(f"Best params: {best_params}")
print(f"Best accuracy: {best_result.val_accuracy:.2f}%")
```

## Installation

```bash
# Install from source
pip install -e .

# Or, to also install MEP optimizers (optional, but recommended)
cd mep && pip install -e . --break-system-packages
```

## Theoretical Foundations

The Bioplausible framework is rooted in several key theoretical principles that drive its design and implementation:

- 🌿 **Local Learning Rules**: Training without global backpropagation. We implement algorithms where weight updates are based on local signals, mirroring biological neural networks and reducing the memory and communication bottlenecks of backpropagation.
- 🌐 **Energy-Based Models (EBMs)**: A significant portion of our algorithms, like Equilibrium Propagation, are grounded in the theory of EBMs, where learning is viewed as shaping an energy landscape.
- 💧 **Event-Driven Computation**: Inspired by the brain's efficiency, our `LazyEqProp` and related models explore event-driven updates, minimizing unnecessary computation.
- 🧬 **Biological Plausibility**: Our algorithms aim to respect biological constraints, such as the need for separate forward and backward pathways (Feedback Alignment) and synaptic plasticity rules (STDP, Hebbian learning).

## Architecture & Algorithms

Bioplausible provides a diverse zoo of models, each exploring a different facet of biologically plausible computation. We are constantly expanding our repertoire.

### Core Equilibrium Propagation (EqProp) Models

The heart of the framework, implementing the foundational EqProp algorithm and its variants.

- **LoopedMLP**: A recurrent MLP that settles into an equilibrium state.
- **BackpropMLP**: A standard feedforward MLP, serving as a crucial baseline for comparison.
- **ConvEqProp**: Convolutional EqProp, showing that the principles can be applied to spatial data.
- **MemoryEfficientLoopedMLP**: Uses gradient checkpointing to enable deeper networks with EqProp.
- **TransformerEqProp**: Applies EqProp dynamics to the Transformer architecture.

### Advanced EqProp Variants

- **NeuralCube**: A 3D lattice topology for embedding computations.
- **LazyEqProp**: Event-driven updates with massive FLOP reduction.
- **FiniteNudgeEP**: Uses a large beta nudge for extreme noise robustness.
- **HolomorphicEP**: Explores complex-valued networks for exact gradients.
- **DirectedEP**: Asymmetric forward and backward weights for deeper scaling.
- **HomeostaticEqProp**: Incorporates biological homeostatic regulation.
- **TemporalResonanceEqProp**: Integrates spike-timing dependent plasticity (STDP).
- **TernaryEqProp**: Uses ternary weights ({-1, 0, +1}) for low-precision computation.
- **SparseEquilibrium**: Enforces top-k sparsity during the settling phase.
- **MomentumEquilibrium**: Adds momentum to the settling process for faster convergence.

### Feedback Alignment Family

A family of algorithms that challenge the weight transport problem of backpropagation.

- **FeedbackAlignmentEqProp**: Uses fixed random feedback weights.
- **AdaptiveFeedbackAlignment**: Allows feedback weights to slowly adapt.
- **DirectFeedbackAlignmentEqProp**: Provides direct feedback from the output layer.
- **StochasticFA**: Introduces noise to the feedback weights for robustness.
- **ContrastiveFeedbackAlignment**: Combines contrastive learning with feedback alignment.
- **EnergyGuidedFA**: Uses energy-based principles to guide the feedback.
- **EnergyMinimizingFA**: Optimizes the energy minimization process.
- **LayerwiseEquilibriumFA**: Achieves equilibrium locally, layer by layer.

### Hebbian & Hybrid Models

- **DeepHebbianChain**: A very deep chain of Hebbian layers, showcasing local learning at scale.
- **ContrastiveHebbianLearning (CHL)**: The precursor to EqProp, based on contrastive learning.
- **PredictiveCodingHybrid**: A hybrid of EqProp and Predictive Coding principles.

### EquiTile Architecture

Our flagship, novel architecture literally interleaved tile-based architecture. It is designed for locality, efficiency, and scalability, and is currently available for several domains.

- **Language**: `EqPropAttentionOnlyLM`, `FullEqPropLM`, `HybridEqPropLM`, `LoopedMLPForLM`, `RecurrentEqPropLM`, `BackpropTransformerLM`, `CausalTransformerEqProp`, `LMEquiTile`, and `FastLMEquiTile`.
- **Vision**: `ConvEquiTile` and various feature extractors.
- **RL**: `RLEquiTile`, `RecurrentRLEquiTile` for reinforcement learning tasks.
- **Graph**: `GraphEquiTile` for graph-structured data.
- **Time-Series**: `TimeSeriesEquiTile` for temporal data.

### Generative & Vision Models

- **EqPropDiffusion**: An energy-based diffusion model.
- **ModernConvEqProp**: A modern, ResNet-style convolutional network.

## Optimization & Training

### Strategy-Pattern Optimizers

Our optimizers are not monolithic; they are built on a flexible strategy pattern, allowing you to compose different gradient processing, update, and feedback strategies into a custom optimizer.

#### Learning Rule Optimizers

- **FeedbackAlignment**: Fixed random feedback weights.
- **DirectFA**: Direct feedback from output.
- **EqProp**: Standard Equilibrium Propagation.
- **HolomorphicEqProp**: Complex-valued EqProp.
- **FiniteNudgeEqProp**: Large beta for noise robustness.
- **LazyEqProp**: Event-driven updates for efficiency.
- **ContrastiveHebbianLearning**: CHL optimizer.

#### MEP Optimizers

A suite of advanced optimizers based on Muon Equilibrium Propagation.

- **smep**: Spectral Muon EP, our default and highest-accuracy optimizer.
- **smep_fast**: A faster variant of SMEP.
- **sdmep**: Low-rank SVD for large models.
- **local_ep**: Layer-local learning.
- **natural_ep**: Natural gradient with Fisher whitening.
- **muon_backprop**: A drop-in SGD replacement.

#### Standard Optimizers

All your familiar friends are also here.

- **SGD**, **Adam**, **AdamW**

### PyTorch Lightning Integration

We provide deep integration with PyTorch Lightning to streamline your research workflows.

#### Lightning Modules

- **BioplausibleLightningModule**: A fully-featured Lightning module that wraps a Bioplausible model. It handles the training loop, validation, logging, and hardware management automatically.
- **BioplausibleStrategy**: A custom Lightning strategy for custom distributed training logic.

### Hyperparameter Optimization

Our `HyperparameterSearch` class uses Optuna to automate the search for the best hyperparameters. This includes:

- **TPE Sampler**: For efficient, Bayesian-like search.
- **NSGA-II Sampler**: For multi-lightweight optimization of multiple objectives (e.g., accuracy and parameter count).
- **Hyperband & Median Pruners**: To cut off unpromising trials early, saving time and compute.
- **Multi-Objective Optimization**: Find the best trade-off between accuracy, model size, and inference speed.

### Training Utilities

- **SupervisedTrainer**: A high-level trainer for standard supervised learning, built on PyTorch Lightning's `Trainer`.
- **RLTrainer**: A trainer for reinforcement learning tasks.
- **BaseTrainer**: A flexible base class for building custom training loops.
- **Training Presets**: A collection of pre-defined configuration presets for common training scenarios.

## Distributed Training & P2P

Bioplausible is built for scale. Train your models on a single GPU, a multi-GPU cluster, or a network of peers.

- **Lightning Trainer**: Seamlessly scale your training with PyTorch Lightning's built-in distributed training capabilities. Just set the `accelerator` and `devices` flags in the `Trainer`.
- **P2P Coordinator**: Our custom P2P system allows for distributed training across multiple nodes with a flexible, Kademlia-based discovery system.
- **Multi-GPU Support**: Our EquiTile architecture supports multi-GPU training with NCCL and asynchronous execution.

## Deployment & Inference

Take your models from research to production.

- **Export**: Export your models to ONNX and TorchScript for deployment on a variety of platforms.
- **Inference Engine**: A high-performance inference engine for loading exported models and making predictions.
- **FastAPI Integration**: Serve your models over a REST API with our built-in FastAPI integration.

## Validation & Testing

Ensuring the correctness and robustness of our algorithms is paramount. Bioplausible includes a comprehensive validation framework.

### Validation Tracks

Our validation system is organized into a suite of specialized tracks:

- **Core Tracks**: Validate the fundamental correctness of the algorithms (e.g., `verify_a_plus_a_equals_2a`).
- **Scaling Tracks**: Test the models under different scaling regimes.
- **Research Tracks**: Evaluate novel ideas and experimental components.
- **Signal Tracks**: Analyze the training dynamics and signal propagation.
- **Honest Tradeoff**: A dedicated track for evaluating the performance-vs-computation tradeoffs.
- **Hardware Tracks**: Performance validation on a variety of hardware platforms.

### Testing

All components of the framework are backed by a comprehensive suite of tests.

```bash
# Run the main test suite
pytest tests/

# Run specific test suites
pytest tests/test_lightning_integration.py
pytest tests/test_mep_integration.py
pytest tests/test_equitile_domains.py
```

## Contributing

The Bioplausible project welcomes contributions!

1. Fork the repository.
2. Create a new branch [branch] for your feature or fix.
3. Make your changes and ensure they are well-tested.
4. Submit a pull request!

Your contributions, big or small, are what make this project thrive. Thank you for helping to advance the field of bioplausible learning!

## Code of Conduct

This project and everyone participating in it is governed by a commitment to respect, collaboration, and constructive communication. We are all here to learn and grow together. Please be kind, patient, and supportive of your fellow contributors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
