# Bioplausible

A comprehensive framework for biologically plausible deep learning, implemented in PyTorch with PyTorch Lightning and Optuna integration. Features a zoo of local-learning algorithms, novel architectures, and automated research tools.

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Architecture & Algorithms](#architecture--algorithms)
- [Optimization & Training](#optimization--training)
- [Automated Research](#automated-research)
- [Distributed Training & P2P](#distributed-training--p2p)
- [Deployment & Inference](#deployment--inference)
- [Validation Framework](#validation-framework)
- [Testing](#testing)
- [Project Structure](#project-structure)

## Overview

Bioplausible explores neural network learning algorithms that operate without global backpropagation. Instead, it implements local learning rules inspired by biological neural networks, including Equilibrium Propagation (EqProp), Feedback Alignment, Hebbian Learning, and tile-based architectures. The framework integrates with PyTorch Lightning for structured training workflows and Optuna for automated hyperparameter optimization.

Local learning algorithms share several properties: synaptic updates depend only on pre-synaptic activity and post-synaptic error, removing the need for symmetric weight transport and reducing memory complexity from O(n) to O(1) per learning unit.

## Features

- 🌿 **Local Learning Algorithms**: Synaptic updates using only local signals
- 🧩 **Tile-Based Architectures**: Partitioned computation units with asynchronous execution
- 🚀 **PyTorch Ecosystem**: PyTorch-first design with full Lightning integration
- 🎯 **Optuna Integration**: Automated hyperparameter search with TPE, NSGA-II, and pruners
- ⚙️ **Strategy-Pattern Optimizers**: Muon Equilibrium Propagation (MEP) optimizers with composable gradient/update strategies
- 🧠 **AutoScientist Agent**: Autonomous experimentalist for continuous algorithm evaluation
- 🔬 **Modular Validation**: Specialized validation tracks for algorithm verification
- 🌐 **Distributed Training**: Multi-GPU support via Lightning and P2P coordinator
- 💻 **Desktop GUI**: PyQt6 interface for experiment management
- 🛠️ **Export Pipelines**: ONNX and TorchScript for model deployment
- 📊 **Analysis Tools**: Statistical analysis and visualization utilities

## Installation

```bash
pip install -e .
cd mep && pip install -e . --break-system-packages
```

## Architecture & Algorithms

### Equilibrium Propagation Family

Algorithms grounded in energy-based models with two-phase dynamics.

- **LoopedMLP**: Recurrent MLP with equilibrium settling
- **BackpropMLP**: Standard feedforward MLP baseline
- **ConvEqProp**: Convolutional EqProp with spectral normalization
- **MemoryEfficientLoopedMLP**: Gradient checkpointing for deep EqProp
- **TransformerEqProp**: EqProp dynamics on Transformer architecture
- **CausalTransformerEqProp**: Autoregressive EqProp transformer
- **EqPropDiffusion**: Energy-based diffusion generative model

### Advanced EqProp Variants

Extensions exploring efficiency, robustness, and biological realism.

- **HolomorphicEP**: Complex-valued networks for exact gradient equivalence
- **DirectedEP**: Asymmetric forward/backward weights for deep scaling
- **FiniteNudgeEP**: Large beta perturbation for finite-difference gradients
- **LazyEqProp**: Event-driven updates reducing redundant computation
- **NeuralCube**: 3D lattice topology with 26-neighbor connectivity
- **TemporalResonanceEqProp**: Spike-timing dependent plasticity (STDP) integration
- **TernaryEqProp**: Low-precision training with {-1, 0, +1} weights
- **SparseEquilibrium**: Top-K sparsity during settling phase
- **MomentumEquilibrium**: Momentum-accelerated settling dynamics
- **HomeostaticEqProp**: Biological homeostatic regulation mechanisms

### Feedback Alignment Family

Solutions to the weight transport problem.

- **AdaptiveFeedbackAlignment**: Slowly-adapting random feedback weights
- **DirectFeedbackAlignmentEqProp**: Direct output-to-hidden feedback with EqProp
- **ContrastiveFeedbackAlignment**: Contrastive learning with feedback signals
- **EnergyGuidedFA**: Energy-based feedback guidance
- **EnergyMinimizingFA**: Energy-minimization feedback alignment
- **LayerwiseEquilibriumFA**: Layer-local equilibrium hybrid
- **EquilibriumAlignment**: Equilibrium-based feedback alignment
- **StochasticFA**: Noisy feedback weights for robustness

### Hebbian Learning Family

Classic and modern local learning implementations.

- **ContrastiveHebbianLearning (CHL)**: Energy-based predecessor to EqProp
- **DeepHebbianChain**: Deep Hebbian layers (500+ layer capability)
- **ThreeFactorHebbian**: Neuromodulated Hebbian with pre x post x reward
- **SpikingSTDP**: Leaky integrate-and-fire with spike-timing plasticity

### Forward-Forward Family

Layer-local goodness-based learning without backward pass.

- **Forward-Forward**: Hinton's layer-local goodness optimization
- **PEPITA**: Present error to perturb input for activity modulation

### Target Propagation Family

Backward target propagation using approximate inverses.

- **DifferenceTargetPropagation**: Target propagation via inverse approximations

### Tile-Based Architectures

Partitioned networks enabling asynchronous, local learning.

- **EquiTile**: Core tile architecture with PC/EP modes
- **FastLMEquiTile**: Transformer-style with Mixture of Tiles sparsity
- **LMEquiTile**: Language modeling tile variants
- **ConvEquiTile**: Vision tile processing
- **RLEquiTile**: Reinforcement learning with tile actor-critic
- **GraphEquiTile**: Graph-structured data with tile message passing
- **TimeSeriesEquiTile**: Temporal forecasting with tile attention

### EqProp Transformer Variants

- **EqPropAttentionOnlyLM**: EqProp in attention layers only
- **FullEqPropLM**: All layers use equilibrium dynamics
- **HybridEqPropLM**: Standard layers + EqProp final layer
- **LoopedMLPForLM**: Recurrent MLP language modeling
- **RecurrentEqPropLM**: Recurrent EqProp for sequences
- **BackpropTransformerLM**: Standard transformer baseline
- **CausalTransformerEqProp**: Autoregressive EqProp transformer

### Vision Models

- **ModernConvEqProp**: ResNet-style CNN optimized for CIFAR-10
- **EqPropDiffusion**: Diffusion generative vision models

### Predictive Coding (FabricPC Integration)

Node-graph topology abstraction and predictive coding training, adapted from [FabricPC](https://github.com/trueagi-io/FabricPC).

- **Graph API:** Define networks as `GraphStructure` with `Linear`/`ReLU`/`Tanh` nodes, `Edge` connections, and `Slot` ports. Validate topology automatically.
- **Dual Training:** Train the same graph with `train_backprop()` (standard autograd) or `train_pcn()` (energy-minimization settling + local weight updates).
- **BioModel Wrapper:** `FabricPCGraphPCN` (`@register_model("fabricpc_graph_pcn")`) integrates with the existing model factory and trainer.
- **No JAX:** Pure PyTorch using `torch.func.grad` for local gradients.

```python
from bioplausible.graph import Linear, ReLU, Edge, TaskMap, graph, initialize_params, train_pcn

inp = Linear(shape=(784, 256), name="input")
act = ReLU(name="hidden")
out = Linear(shape=(256, 10), name="output")
g = graph(nodes=[inp, act, out],
    edges=[Edge(inp, act.slot("input")), Edge(act, out.slot("input"))],
    task_map=TaskMap(x=inp, y=out))

params = initialize_params(g)
results = train_pcn(g, params, train_loader, epochs=5)
```

See `examples/fabricpc_mnist_bridge.py` and `FABRICPC_INTEGRATION.md` for details.

## Optimization & Training

### Learning Rule Optimizers

Local-update rule implementations.

- **FeedbackAlignment**: Fixed random feedback signal
- **DirectFA**: Direct output feedback pathway
- **EqProp**: Standard two-phase equilibrium propagation
- **HolomorphicEqProp**: Complex-valued gradient equivalence
- **FiniteNudgeEqProp**: Large beta perturbation
- **LazyEqProp**: Event-driven updates
- **ContrastiveHebbianLearning**: Contrastive local update

### MEP Optimizers

Muon-based equilibrium propagation with strategy composition.

- **smep**: Spectral Muon Equilibrium Propagation
- **smep_fast**: Optimized SMEP variant
- **sdmep**: Low-rank SVD for large model scaling
- **local_ep**: Layer-local learning
- **natural_ep**: Natural gradient with Fisher whitening
- **muon_backprop**: Muon optimizer combined with backprop

### Standard Optimizers

- **SGD**, **Adam**, **AdamW**: PyTorch baseline optimizers

### PyTorch Lightning Integration

Structured training workflows with automatic hardware management.

- **BioLightningModule**: Lightning module wrapping Bioplausible models
- **BioOptunaPruner**: Optuna pruning callback for early stopping
- **BioRayTuneSearch**: Ray Tune hyperparameter search integration
- **BioPrecisionCallback**: Automatic mixed precision support
- **EnergyConvergenceCallback**: EqProp-specific convergence monitoring
- **BioPredictionWriter**: Prediction output callback
- **BioPrecisionMixin**: Mixin for precision-aware modules
- **run_pl_trial**: Single-trial Lightning execution
- **run_pl_trial_with_wandb**: WandB-integrated trial execution
- **run_nas_search**: Neural architecture search integration
- **build_trainer**: Configured Lightning Trainer builder

## Automated Research

### AutoScientist Agent

Autonomous experimental loop managing resource allocation and strategy selection.

```
                    ┌──────────────────┐
                    │  Initialize Task │
                    │ (Strategy + ID)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼────────┐    ┌─────────────────┐
                    │   Checkpoint?   │──No▶│ Launch Training │
                    └────────┬─────────┘    └────────┬────────┘
                             │                      ▼
                            Yes          ┌────────────────────┐
                             │          │ Monitor Resources    │
                             ▼          │ & Training Dynamics  │
                    ┌─────────────────┐ └────────────────────┘
                    │ Load State      │           │
                    └─────────────────┘           ▼
                             │          ┌────────────────────┐
                             └─────────▶│ Training Complete? │
                                        └──────────┬───────────┘
                                                   │
                                                  No
                                           ┌───────┴───────┐
                                           ▼               ▼
                                 ┌─────────────────┐  ┌──────────────┐
                                 │ Save Checkpoint │  │Continue Loop │
                                 └───────┬─────────┘  └──────────────┘
                                         │
                                         ▼
                               ┌──────────────────┐
                               │ Analyze Results  │
                               └───────┬──────────┘
                                       │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         ┌────────────────────┐                    ┌──────────────────────┐
         │ Select Next        │                    │ Decision: Promote or │
         │ Strategy           │◀─────────────────┤ Explore Alternative  │
         │ (Smoke/Shallow/Deep)│                    └──────────────────────┘
         └────────────────────┘
                    │
                    ▼
         ┌────────────────────┐
         │ Generate Experiment│
         │ Tasks              │
         └────────────────────┘
```

### Hyperparameter Search

Optuna-powered search with multiple samplers and pruners.

- **TPE Sampler**: Tree-structured Parzen estimator for Bayesian optimization
- **NSGA-II Sampler**: Multi-objective Pareto front optimization
- **Hyperband Pruner**: Successive halving for early stopping
- **Median Pruner**: Statistical pruning based on intermediate results

### Experiment Runner

- **ExperimentRunner**: Standardized evaluation across models/optimizers
- **HyperparameterSearch**: Grid and random search interfaces
- **quick_comparison**: Side-by-side algorithm comparison
- **benchmark_model**: Performance benchmarking utilities

## Distributed Training & P2P

### PyTorch Lightning Scaling

Multi-GPU and multi-node training via Lightning Trainer.

- **accelerator**: "gpu", "tpu", or "cpu"
- **devices**: Number of devices per node
- **strategy**: "ddp", "fsdp", "deepspeed" for distributed

### P2P Coordinator System

Decentralized training coordination with Kademlia discovery.

```
┌────────────────────┐
│   Coordinator      │
│ (Task Dispatcher)  │
└─────────┬───────────┘
          │
     ┌────▼────┬─────────────┬─────────────┐
     │         │             │             │
     ▼         ▼             ▼             ▼
┌───────┐ ┌───────┐     ┌───────┐     ┌───────┐
│Worker1 │ │Worker2 │ ... │WorkerN │ ... │WorkerM │
│(Learner)││(Learner)│   │(Learner)│   │(Learner)│
└───────┘ └───────┘     └───────┘     └───────┘
     │         │             │             │
     └─────────┼─────────────┼─────────────┘
               ▼
┌────────────────────────────┐
│ Distributed Training         │
│ Progress Aggregation       │
└────────────────────────────┘
```

### EquiTile Parallelism

Asynchronous tile execution across devices.

- **Tile-Parallel Scheduling**: No inter-tile synchronization barriers
- **NCCL Backend**: High-speed GPU communication
- **Dynamic Tile Growth**: Runtime tile addition/removal

## Deployment & Inference

### Model Export

Serialization for production deployment.

- **ONNX Export**: Cross-platform model format
- **TorchScript Export**: C++/Python runtime compatibility
- **Quantization**: INT8 and ternary weight support

### Inference Engine

Runtime prediction interface.

- **InferenceEngine**: High-throughput prediction server
- **FastAPI Endpoint**: REST API for model serving
- **Batch Processing**: Optimized input batching

## Validation Framework

### Validation Tracks

Specialized evaluation protocols organized by focus area.

- **Core Tracks**: Fundamental algorithm correctness verification
- **Scaling Tracks**: Depth and width scaling behavior analysis
- **Research Tracks**: Experimental algorithm evaluation
- **Signal Tracks**: Training dynamics and signal propagation
- **Honest Tradeoff**: Performance vs. computation cost evaluation
- **Hardware Tracks**: Cross-platform performance validation
- **Application Tracks**: Domain-specific benchmarking
- **Architecture Comparison**: Model-to-model performance comparisons
- **Negative Results**: Documentation of unsuccessful approaches
- **NEBC Tracks**: Novelty, Efficiency, Biological Plausibility, and Correctness assessment

### Analysis Tools

- **ResultAnalyzer**: Statistical analysis of experiment results
- **TrainingVisualizer**: Loss curves and convergence plots
- **ScalingAnalyzer**: Model scaling behavior characterization
- **FailureManifesto**: Negative result documentation
- **AblationAnalyzer**: Component ablation studies

### Hyperparameter Optimization Flow

```
┌────────────────────────────────────┐
│          Experiment Start          │
│   (Model + Optimizer + Params)     │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│        Optuna Trial Scheduler      │
│   (Selects params from search)     │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│     Patience-Level Validation      │
│  SMOKE → SHALLOW → STANDARD → …    │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│     Training Callback Hooks        │
│ (Energy Convergence, Precision, …) │
└───────────────┬────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│      Trial Pruning Decision        │
│   (Intermediate metrics → prune?)  │
└───────────────┬────────────────────┘
                │
          Continue/Prune
                │
                ▼
┌────────────────────────────────────┐
│      Parameter Update Decision     │
│ (Next iteration of search space)   │
└────────────────────────────────────┘
```
