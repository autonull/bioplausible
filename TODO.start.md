# TODO.start.md — Phases 0 & 1: The Definitive Implementation Plan
**Bioplausible · Weeks 1–10 · From Prototype to Platform**

*This document translates the strategic vision in TODO.md into concrete, actionable engineering tasks grounded in the current codebase. Every item references exact files, functions, and APIs. Every algorithm is chosen for its potential to produce a result that no other open-source framework can match.*

---

## Why Bioplausible Must Exist

The ML ecosystem has PyTorch, JAX, and hundreds of training frameworks — all built around backpropagation. There is **no** unified, open-source platform that:

1. **Treats bio-plausible and alternative learning rules as first-class citizens** alongside backprop baselines, on equal footing across all domains.
2. **Provides a composable optimizer algebra** (gradient strategy × update strategy × constraint strategy) that lets researchers mix-and-match EP, Muon, Dion, FA, Hebbian, PC, and Forward-Forward components.
3. **Generates a self-improving knowledgebase** that distills *why* configurations succeed or fail — not just leaderboard numbers.
4. **Maps naturally to neuromorphic hardware** — local learning rules, event-driven dynamics, and O(1) memory training that backprop frameworks cannot offer.

This plan makes that vision real in 10 weeks on commodity hardware.

---

## Codebase Audit: What We Have vs. What We Need

### ✅ Existing Strengths (Leverage, Don't Rebuild)

| Asset | Location | Depth |
|---|---|---|
| **MEP Optimizer Suite** | `mep/mep/optimizers/` | SMEP, SDMEP, LocalEP, NaturalEP, MuonUpdate, DionUpdate, SpectralConstraint, ErrorFeedback, O(1) memory (v1+v2), EWC continual learning — *composable strategy pattern* |
| **Model Zoo** (30+ algorithms) | `bioplausible/models/` | EqProp MLP/Conv/Transformer/Holomorphic/Directed/FiniteNudge/Diffusion, FA/DFA/StochasticFA, CHL, DeepHebbian, PredictiveCoding, SparseEq, MomentumEq, NeuralCube, EquiTile (5 variants), TileEQ/ATPC (2600 lines) |
| **Optimizer Registry** | `bioplausible/optimizers/` | 18 optimizers including MEP presets (`smep`, `sdmep`, `local_ep`, `natural_ep`, `muon_backprop`) |
| **SupervisedTrainer** | `bioplausible/training/supervised.py` | 764 lines: vision, LM, RL dispatch; kernel mode; safety wrapper; compile; schedulers |
| **RL Trainer** | `bioplausible/training/rl.py` | CartPole, Pendulum, Acrobot |
| **Task Abstractions** | `bioplausible/hyperopt/tasks.py` | VisionTask, LMTask, RLTask, CharNGramTask, BaseTask; K-fold CV; data caching |
| **Datasets** | `bioplausible/datasets.py` | MNIST, CIFAR-10/100, FashionMNIST, SVHN, KMNIST, USPS, digits; Shakespeare/WikiText-2/PTB (char-level) |
| **Scientist Module** | `bioplausible/scientist/` | Strategy, synthesizer, failure tracker, archiver, robustness, training dynamics, algorithm constraints |
| **Hyperopt** | `bioplausible/hyperopt/` | Optuna bridge, metamodel, eval tiers, parallel runner, search space, experiment runner |
| **Validation Framework** | `bioplausible/validation/` | 20 validation tracks, scientific rigor checks |
| **PyQt6 UI** | `bioplausible_ui/` | Dashboard, lab, model builder |
| **Package** | `pyproject.toml` | pip-installable with entry points |

### ❌ Critical Gaps (Phase 0–1 Targets)

| # | Gap | Impact | Solution |
|---|---|---|---|
| 1 | **Missing SOTA algorithm families** (Forward-Forward, PEPITA, Target Propagation, Spiking/STDP, Three-Factor rules) | Can't claim comprehensive coverage | Implement 5 new families |
| 2 | **MEP not integrated into main training loop** | Most powerful optimizer isolated in separate package | Bridge `mep` into `SupervisedTrainer` via unified runner |
| 3 | **No energy/compute metrics** per run | Can't produce energy-efficiency comparisons | Add FLOPs/sparsity/wall-time tracking |
| 4 | **No declarative config-driven runner** | Can't do "one YAML → run anything" | OmegaConf `RunConfig` + `runner.py` |
| 5 | **No graph/tabular domain** | Missing two major ML verticals | `GraphTask` + `TabularTask` |
| 6 | **No knowledgebase seed** (GP surrogate) | Scientist module has no predictive model | `knowledge/` subpackage |
| 7 | **No systematic ablation framework** | Ablation scripts scattered as ad-hoc files | `AblationStudy` class |
| 8 | **Char-level only LM tokenization** | Can't fairly compare on WikiText-2 | Add BPE tokenizer option |
| 9 | **No scaling-law analysis** | Can't extrapolate to justify compute requests | `analysis/scaling.py` |
| 10 | **Version mismatch** | `pyproject.toml` = 0.1.0, `__init__.py` = 0.2.0 | Synchronize |

---

## Algorithm Roster: The Complete Competitive Lineup

This is the definitive set of algorithm families for Phase 1 experiments. Each is chosen because it represents a **distinct credit-assignment paradigm** — the point is not to have the most models, but to exhaustively cover the *space of learning mechanisms*.

### Tier 1 — Flagship (Maximize Investment)

These are algorithms where Bioplausible has or can build a **unique competitive advantage**.

| # | Family | Key Algorithms | Credit Assignment | Why It Matters | Existing? |
|---|---|---|---|---|---|
| 1 | **Muon EqProp (MEP)** | SMEP, SDMEP, LocalEP, NaturalEP | Equilibrium contrast + Muon orthogonalization + Dion low-rank | **Our crown jewel.** No other framework has EP + geometry-aware optimization. O(1) memory enables impossibly deep networks. | ✅ `mep/` |
| 2 | **ATPC / TileEQ** | Adaptive Tile-Based Predictive Coding | Local prediction errors on tile graphs | **Neuromorphic-native.** Event-driven, dynamic topology, hardware-abstractable. Maps to memristors/optical chips. | ✅ `tile_eq.py` |
| 3 | **EquiTile** | EquiTile, EquiTile-EP, ConvEquiTile, LM-EquiTile, RL-EquiTile | Tiled local learning with optional EP | **Scalable local learning.** 5 domain variants already built. | ✅ `equitile/` |

### Tier 2 — Established (Strong Existing Implementations)

| # | Family | Key Algorithms | Credit Assignment | Existing? |
|---|---|---|---|---|
| 4 | **Standard EqProp** | EqProp MLP, Holomorphic EP, Directed EP, Finite-Nudge EP, Conv EqProp | Equilibrium contrast (symmetric/asymmetric) | ✅ |
| 5 | **Feedback Alignment** | FA, DFA, Adaptive FA, Stochastic FA, Energy-Guided FA, Energy-Minimizing FA | Random fixed feedback weights | ✅ |
| 6 | **Contrastive Hebbian** | CHL, Deep Hebbian (100-layer) | Pre/post synaptic correlation contrast | ✅ |
| 7 | **Predictive Coding** | PC-Hybrid | Top-down prediction error minimization | ✅ |
| 8 | **Backprop Baseline** | Backprop MLP, Backprop Transformer | Full gradient chain (the control group) | ✅ |

### Tier 3 — New Implementations (Phase 0–1 Additions)

These fill critical gaps in the learning-rule taxonomy. Each represents a paradigm **not yet covered**.

| # | Family | Credit Assignment | Why Add It | Complexity |
|---|---|---|---|---|
| 9 | **Forward-Forward (Hinton 2022)** | Layer-local goodness maximization (positive) / minimization (negative) | **No backward pass at all.** The most radical departure from backprop. Extremely neuromorphic-friendly. | Medium |
| 10 | **PEPITA** | Error-driven input modulation → two forward passes | **Forward-only + top-down feedback.** Proven equivalent to Forward-Forward under certain conditions. Memory-efficient. | Medium |
| 11 | **Difference Target Propagation** | Layer-wise targets via learned approximate inverses | **Targets not gradients.** Connected to Gauss-Newton optimization. Scales better than vanilla TP. | Medium-High |
| 12 | **Three-Factor Hebbian / Neuromodulated** | pre × post × neuromodulator signal | **Dopamine-like global reward signal.** Bridges Hebbian learning with RL. Biologically most realistic for reward-based tasks. | Medium |
| 13 | **Spiking + STDP** | Spike-Timing-Dependent Plasticity | **Temporal credit assignment.** Essential for neuromorphic hardware claims. Event-driven computation. | High |

### Algorithm Count Summary

- **Existing implementations:** 30+ model variants across 8 families
- **New implementations:** 5 new families (Forward-Forward, PEPITA, DTP, Three-Factor, STDP)
- **Total Phase 1 coverage:** **13 distinct credit-assignment paradigms**
- **Unique optimizers:** 18 (including 6 MEP variants)

This is **unmatched** by any comparable framework.

---

## Phase 0: Unified Distillation (Weeks 1–4)

> **Milestone:** `pip install -e bioplausible && python examples/cross_domain_demo.py` runs any algorithm on any domain from a single YAML config.

### 0.1 — Package Cleanup and Version Sync

**Files:** `pyproject.toml`, `bioplausible/__init__.py`

- [ ] Synchronize version to `0.3.0` (significant new capabilities warrant a minor bump).
- [ ] Update `description` to: *"Unified platform for exploring the full spectrum of learning rules across machine intelligence"*.
- [ ] Add optional dependency groups:
  ```toml
  [project.optional-dependencies]
  knowledgebase = ["gpytorch", "botorch", "sympy"]
  graphs = ["torch-geometric>=2.5", "networkx"]
  spiking = ["snnTorch>=0.8"]
  full = ["bioplausible[knowledgebase,graphs,spiking]"]
  ```
- [ ] Add `omegaconf>=2.3` to core dependencies.
- [ ] Export `list_model_specs()` returning `List[ModelSpec]` from `__init__.py`.

---

### 0.2 — Enhanced Registry with Compute Metadata

**File:** `bioplausible/models/registry.py`

Extend `ModelSpec` to support the knowledgebase and energy tracking:

```python
@dataclass
class ModelSpec:
    ...
    # New fields for Phase 0
    credit_locality: str = "global"        # "global" | "local" | "layerwise" | "forward-only" | "equilibrium"
    domain_support: List[str] = field(default_factory=lambda: ["vision"])
    param_count_approx: Optional[int] = None
    training_paradigm: str = "supervised"  # "supervised" | "self-supervised" | "rl" | "hybrid"
    learning_rule_class: str = "gradient"  # "gradient" | "equilibrium" | "hebbian" | "target" | "forward-only" | "spiking"
    hardware_affinity: List[str] = field(default_factory=list)  # ["gpu", "neuromorphic", "optical", "memristor"]
    requires_backward: bool = True         # False for EP, FF, PEPITA, Hebbian, STDP
    memory_complexity: str = "O(N)"        # "O(1)" for MEP O(1) variants, "O(N)" standard
```

- [ ] Populate all 30+ existing `ModelSpec` entries with these new fields.
- [ ] Add specs for the 5 new algorithm families (Forward-Forward, PEPITA, DTP, Three-Factor, STDP).

---

### 0.3 — Declarative Config System

**New file:** `bioplausible/config_schema.py`

OmegaConf-based config with structured typing:

```python
@dataclass
class RunConfig:
    seed: int = 42
    device: str = "auto"                   # "auto" selects cuda if available
    output_dir: str = "results/${now:%Y%m%d_%H%M%S}"

    @dataclass
    class Data:
        task: str = MISSING                # "mnist", "cifar10", "shakespeare", "cartpole", "cora"
        batch_size: int = 64
        seq_len: int = 64                  # LM tasks
        augment: bool = False
        data_fraction: float = 1.0         # for data-efficiency curves

    @dataclass
    class Model:
        name: str = MISSING                # registry key
        hidden_dim: int = 256
        num_layers: int = 3
        extra: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Optimizer:
        name: str = "adam"                 # any key from OPTIMIZER_REGISTRY
        lr: float = 0.001
        weight_decay: float = 0.0
        # MEP-specific
        beta: float = 0.5
        settle_steps: int = 30
        mode: str = "ep"                   # "ep" | "backprop"

    @dataclass
    class Trainer:
        epochs: int = 10
        batches_per_epoch: int = 100
        grad_clip: Optional[float] = None
        scheduler: Optional[str] = None
        use_compile: bool = True
        track_energy: bool = True

    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    trainer: Trainer = field(default_factory=Trainer)
    ablation_tags: Dict[str, Any] = field(default_factory=dict)
```

**New file:** `bioplausible/runner.py`

```python
def run_from_config(cfg: RunConfig) -> Dict[str, Any]:
    """
    The universal entry point.

    1. Seeds everything.
    2. Resolves task via create_task(cfg.data.task).
    3. Creates model via create_model(cfg.model.name).
    4. Creates optimizer via create_optimizer(model, cfg.optimizer.name).
       - For MEP optimizers: passes beta, settle_steps, mode etc.
       - For standard optimizers: passes lr, weight_decay.
    5. Builds SupervisedTrainer (or RLTrainer).
    6. Runs training with energy tracking.
    7. Returns metrics dict + saves structured JSON log.
    """
```

This is the **critical integration point** that bridges the isolated `mep/` module into the main training harness. The runner must handle:
- MEP optimizers that call `optimizer.step(x=x, target=y)` (no `.backward()`).
- Standard optimizers that expect `loss.backward(); optimizer.step()`.
- Forward-only algorithms (FF, PEPITA) with dual forward passes.
- Hebbian/STDP rules with local weight updates (no global loss).

---

### 0.4 — Energy-Proxy Metrics

**New file:** `bioplausible/energy.py`

```python
@dataclass
class EnergyProfile:
    forward_flops: int          # via torch.profiler or hook counting
    backward_flops: int         # 0 for EP/FF/PEPITA/Hebbian
    param_count: int
    activation_sparsity: float  # fraction of near-zero activations
    weight_sparsity: float      # fraction of near-zero weights
    wall_time_ms: float         # elapsed per batch
    peak_memory_mb: float       # torch.cuda.max_memory_allocated
    energy_proxy: float         # (fwd + bwd flops) × (1 - activation_sparsity) / param_count
    requires_backward: bool     # from ModelSpec — crucial for neuromorphic comparison

def profile_run(model, input_shape, n_batches=10) -> EnergyProfile: ...
```

**Integration:**
- [ ] Add `EnergyTracker` context manager in `SupervisedTrainer.train_batch()`.
- [ ] Log `EnergyProfile` fields in `train_epoch()` results.
- [ ] **Key insight:** Report `backward_flops = 0` for EP, FF, PEPITA, Hebbian, STDP families. This is the quantitative argument for neuromorphic advantage.

---

### 0.5 — Structured Logging and Ablation Hooks

**File:** `bioplausible/training/supervised.py`

- [ ] Add `log_ablation(tag: str, value: Any)` method.
- [ ] Add `ablation_tags` parameter to constructor (from `RunConfig`).
- [ ] On each epoch, emit structured JSON line to `{output_dir}/runs.jsonl`:
  ```json
  {
    "epoch": 3, "model": "smep_eqprop_mlp", "task": "mnist",
    "optimizer": "smep", "lr": 0.01, "beta": 0.5,
    "val_accuracy": 0.954, "val_loss": 0.18,
    "forward_flops": 2400000, "backward_flops": 0,
    "energy_proxy": 1440000, "wall_time_ms": 82,
    "peak_memory_mb": 210, "requires_backward": false,
    "tags": {"eq_steps": 30, "spectral_bound": 0.95}
  }
  ```

**File:** `bioplausible/tracking.py`
- [ ] Add `log_config(cfg: dict)` and `log_energy(profile: EnergyProfile)`.

---

### 0.6 — New Algorithm Implementations

#### 0.6.1 — Forward-Forward Algorithm

**New file:** `bioplausible/models/forward_forward.py`

```python
@register_model("forward_forward")
class ForwardForwardNet(BioModel):
    """
    Hinton's Forward-Forward (2022).
    Two forward passes (positive/negative), layer-local goodness objective.
    No backward pass. requires_backward = False.

    Positive pass: real data with correct label embedded → maximize sum of squared activations.
    Negative pass: corrupted data or wrong label → minimize sum of squared activations.
    """
    def train_step(self, x_pos, x_neg):
        for layer in self.layers:
            # Positive: maximize goodness
            h_pos = layer(x_pos)
            g_pos = (h_pos ** 2).sum(dim=1).mean()

            # Negative: minimize goodness
            h_neg = layer(x_neg)
            g_neg = (h_neg ** 2).sum(dim=1).mean()

            # Layer-local loss
            loss = -torch.log(torch.sigmoid(g_pos - threshold)) \
                   - torch.log(torch.sigmoid(threshold - g_neg))
            loss.backward()  # local backward, not through whole network
            layer.optimizer.step()
            layer.optimizer.zero_grad()

            # Detach for next layer (no gradient flow between layers)
            x_pos = h_pos.detach()
            x_neg = h_neg.detach()
```

#### 0.6.2 — PEPITA

**New file:** `bioplausible/models/pepita.py`

```python
@register_model("pepita")
class PEPITA(BioModel):
    """
    PEPITA: Present the Error to Perturb the Input To modulate Activity.
    Two forward passes; error-modulated input; no backward pass through network.
    """
    def train_step(self, x, y):
        # Pass 1: standard forward
        h_standard = self.forward(x)
        error = compute_error(h_standard, y)

        # Modulate input with error
        x_mod = x + self.feedback_matrix @ error

        # Pass 2: modulated forward
        h_modulated = self.forward(x_mod)

        # Update: Δw ∝ (h_modulated - h_standard) × input
        for layer, (h_s, h_m) in zip(self.layers, zip(standards, modulateds)):
            delta = h_m - h_s
            layer.weight += self.lr * (delta.T @ layer.input)
```

#### 0.6.3 — Difference Target Propagation

**New file:** `bioplausible/models/target_prop.py`

```python
@register_model("diff_target_prop")
class DifferenceTargetProp(BioModel):
    """
    Difference Target Propagation (Lee et al. 2015).
    Propagates targets (not gradients) backward using learned approximate inverses.
    Linear correction for imperfect autoencoders.
    """
    # Each layer learns:
    # - forward: f_i(h_{i-1}) → h_i
    # - inverse: g_i(h_i) → ĥ_{i-1}  (autoencoder)
    # Target for layer i: t_i = h_i + g_{i+1}(t_{i+1}) - g_{i+1}(h_{i+1})
```

#### 0.6.4 — Three-Factor Neuromodulated Hebbian

**New file:** `bioplausible/models/three_factor.py`

```python
@register_model("three_factor_hebbian")
class ThreeFactorHebbian(BioModel):
    """
    Three-Factor Learning: Δw = η · M · pre · post
    where M is a neuromodulatory signal (dopamine-like global reward).

    Bridges Hebbian learning with RL.
    Suitable for episodic tasks with sparse reward signals.
    """
```

#### 0.6.5 — Spiking STDP (Lightweight)

**New file:** `bioplausible/models/spiking_stdp.py`

```python
@register_model("spiking_stdp")
class SpikingSTDP(BioModel):
    """
    Leaky Integrate-and-Fire neurons with Spike-Timing-Dependent Plasticity.
    Uses snnTorch for LIF dynamics; custom STDP learning rule overlaid.

    Fully event-driven: computation only occurs on spikes.
    requires_backward = False, hardware_affinity = ["neuromorphic"]
    """
```

---

### 0.7 — Domain Expansion

#### Graph Tasks

**New file:** `bioplausible/hyperopt/graph_task.py`

```python
class GraphTask(BaseTask):
    """Node classification on Cora / Citeseer / PubMed via torch-geometric."""
    task_type = "graph"
    # Uses GCN message-passing; compatible with EqProp (equilibrium on node embeddings),
    # Hebbian (local Hebb updates on edges), and Forward-Forward (node goodness).
```

**New file:** `bioplausible/models/graph_eqprop.py`

```python
@register_model("graph_eqprop")
class GraphEqProp(BioModel):
    """GCN backbone with EqProp equilibrium settling on node embeddings."""
```

Register graph datasets: Cora, Citeseer, PubMed.

#### Tabular Tasks

**New file:** `bioplausible/hyperopt/tabular_task.py`

```python
class TabularTask(BaseTask):
    """UCI / OpenML tabular classification. Lightweight, fast iteration."""
    task_type = "tabular"
```

Add in `create_task()`: Iris, Wine, Breast Cancer (sklearn).

---

### 0.8 — Knowledgebase Seed

**New directory:** `bioplausible/knowledge/`

**New file:** `bioplausible/knowledge/seed.py`

```python
class KnowledgebaseSeed:
    """
    GP surrogate fitted to completed experiments.
    Features: one-hot(model_family, task_type) + log(lr) + num_layers + hidden_dim.
    Target: val_accuracy.
    Uses GPyTorch ExactGP with MaternKernel.
    """
    def fit(self, db_path="bioplausible.db"): ...
    def predict(self, config: Dict) -> Tuple[float, float]: ...
    def top_k(self, task: str, k: int = 5) -> List[Dict]: ...
```

---

### 0.9 — Example Configs and Cross-Domain Demo

**New directory:** `configs/`

```
configs/
  mep_mnist.yaml              # MEP (SMEP) on MNIST — flagship demo
  eqprop_mnist.yaml           # Standard EqProp on MNIST
  forward_forward_mnist.yaml  # New: FF on MNIST
  backprop_mnist.yaml         # Baseline
  eqprop_shakespeare.yaml     # LM
  rl_cartpole.yaml            # RL
  sweep_phase1.yaml           # Full Phase 1 sweep template
```

**New file:** `examples/cross_domain_demo.py`

Demonstrates same runner, same config schema, five paradigms:
1. MEP (SMEP) on MNIST — no backward pass, O(1)-memory capable
2. Forward-Forward on MNIST — no backward pass, layer-local
3. Backprop on MNIST — baseline control
4. EqProp Transformer on Shakespeare — LM
5. EquiTile on CartPole — RL

---

### Phase 0 Completion Criteria

- [ ] `pip install -e ".[full]"` succeeds.
- [ ] `python examples/cross_domain_demo.py` runs 5 paradigms × 3 domains without errors.
- [ ] `pytest tests/ -x -q` passes with no regressions.
- [ ] Every run logs `EnergyProfile` including `backward_flops` and `requires_backward`.
- [ ] `knowledge/surrogate.pt` exists after seeding from existing DB.
- [ ] All 5 new model files (`forward_forward.py`, `pepita.py`, `target_prop.py`, `three_factor.py`, `spiking_stdp.py`) pass their unit tests.

---

## Phase 1: Commodity-Hardware Ignition (Weeks 5–10)

> **Goal:** Produce an arXiv-quality technical report demonstrating decisive preliminary signals across 13 algorithm families on 6+ domains. Deliver the first interactive knowledgebase.

### 1.1 — The Experiment Matrix

A **controlled, publication-quality** sweep. Every cell is a `RunConfig` YAML.

#### Algorithm × Domain Grid

| # | Algorithm | MNIST | CIFAR-10 | Shakespeare | CartPole | Cora | UCI-Iris |
|---|---|---|---|---|---|---|---|
| 1 | **SMEP** (Muon EqProp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | **SDMEP** (+ Dion low-rank) | ✅ | ✅ | ✅ | | | |
| 3 | **LocalEP** (layer-local EP) | ✅ | ✅ | | | ✅ | |
| 4 | **NaturalEP** (Fisher info) | ✅ | | | | | |
| 5 | Standard EqProp MLP | ✅ | | | ✅ | | ✅ |
| 6 | Holomorphic EqProp | ✅ | | | | | |
| 7 | Conv EqProp | | ✅ | | | | |
| 8 | Forward-Forward | ✅ | ✅ | | | | ✅ |
| 9 | PEPITA | ✅ | ✅ | | | | |
| 10 | Diff Target Prop | ✅ | ✅ | | | | |
| 11 | DFA | ✅ | ✅ | | ✅ | | |
| 12 | CHL | ✅ | | | ✅ | | ✅ |
| 13 | Deep Hebbian (100L) | ✅ | | | | | |
| 14 | Three-Factor Hebbian | ✅ | | | ✅ | | |
| 15 | PC Hybrid | ✅ | | | | | |
| 16 | EquiTile | ✅ | | | ✅ | | |
| 17 | EquiTile-EP | ✅ | | | | | |
| 18 | ATPC/TileEQ | ✅ | | | | | |
| 19 | Spiking STDP | ✅ | | | | | |
| 20 | **Backprop Baseline** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Total unique configs:** ~80 | **Seeds per config:** 3 | **Total runs:** ~240

#### Controlled Variables

| Variable | Value | Rationale |
|---|---|---|
| Seeds | `{0, 1, 2}` | 3 replicas for error bars |
| Budget per run | 10 epochs × 200 batches | Standardized compute |
| VRAM limit | ≤ 6 GB | Consumer GPU (RTX 3060 tier) |
| Wall time limit | ≤ 2 hours per run | Practical ceiling |
| LR grid | `{1e-4, 3e-4, 1e-3, 3e-3}` | Report best per algorithm |

---

### 1.2 — The Five Decisive Signals

Phase 1 must produce results that **no other framework can produce.** Focus on these five signals:

#### Signal 1: Backward-Free Parity
> *"Multiple backward-free algorithms (MEP, FF, PEPITA) match or exceed backprop on MNIST/CIFAR-10 within the same compute envelope."*

- Compare `requires_backward=False` family accuracy vs. backprop baseline.
- Report `backward_flops=0` explicitly.
- **Table 1** in the report.

#### Signal 2: Energy Efficiency Frontier
> *"EP and local-learning methods achieve the same accuracy at 30–60% lower energy proxy on MNIST."*

- Plot accuracy vs. energy_proxy (Pareto frontier).
- Highlight algorithms with high accuracy AND low energy.
- **Figure 1** (Pareto plot).

#### Signal 3: Data Efficiency Advantage
> *"Local learning rules require 2–5× less data than backprop to reach 90% accuracy on reduced MNIST."*

- Ablation: data_fraction ∈ `{0.01, 0.05, 0.1, 0.25, 0.5, 1.0}`.
- Compare accuracy-at-fraction curves.
- **Figure 2** (data efficiency curves).

#### Signal 4: Depth Scaling Without Backward
> *"MEP with O(1) memory trains 100+ layer networks where standard EqProp fails."*

- Depth sweep: `{4, 8, 16, 32, 64, 128}` layers.
- Report which algorithms converge vs. diverge at each depth.
- Highlight `O1MemoryEP` variants.
- **Table 2** (depth × convergence matrix).

#### Signal 5: Cross-Domain Generality
> *"The same EquiTile architecture trains vision, LM, and RL without per-domain customization."*

- Run EquiTile variants across all 3 major domains with identical hyperparams.
- Compare against domain-specific baselines.
- **Table 3** (cross-domain generality).

---

### 1.3 — Ablation Study Framework

**New file:** `bioplausible/analysis/ablation.py`

```python
class AblationStudy:
    """
    Systematic parameter sensitivity study.

    Dimensions (each run varies one dimension, holds others at default):
    1. Learning rate:     [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    2. Model depth:       [2, 4, 6, 8]
    3. Hidden dim:        [64, 128, 256, 512]
    4. EqProp eq_steps:   [5, 10, 20, 40]      (EqProp family only)
    5. EqProp beta:       [0.01, 0.1, 0.5, 1.0] (EqProp family only)
    6. Muon NS iters:     [3, 5, 7]             (MEP only)
    7. Sparsity target:   [0.0, 0.5, 0.7, 0.9]  (Sparse models only)
    8. Data fraction:     [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    9. Spectral bound γ:  [0.8, 0.9, 0.95, 0.99] (spectrally constrained models)
    """
    def __init__(self, base_cfg: RunConfig, dimensions: Dict[str, List]): ...
    def run(self, parallel_workers=4) -> pd.DataFrame: ...
    def plot_sensitivity_heatmap(self, metric="val_accuracy") -> plt.Figure: ...
    def identify_critical_hyperparams(self) -> List[str]: ...
```

---

### 1.4 — Scaling Law Analysis

**New file:** `bioplausible/analysis/scaling.py`

```python
def fit_power_law(param_counts: List[int], losses: List[float]) -> Tuple[float, float]:
    """Fit L = a · N^(-b). Return (a, b). Positive b means loss decreases with scale."""

def compute_compute_optimal(results_df: pd.DataFrame) -> pd.DataFrame:
    """For each algorithm, find the compute-optimal model size (Chinchilla-style)."""

def plot_scaling_curves(results_df: pd.DataFrame) -> plt.Figure:
    """Per-algorithm scaling curves with power-law fits and confidence bands."""
```

Parameter count sweep: `[10k, 30k, 100k, 300k, 1M]` via `(hidden_dim, num_layers)` combinations.

---

### 1.5 — Failure Modes Manifesto

**Extension of:** `bioplausible/scientist/failure_tracker.py`

New failure categories:
```python
class FailureCategory(Enum):
    CONVERGENCE_FAILURE = "convergence_failure"
    GRADIENT_EXPLOSION = "gradient_explosion"
    SETTLING_DIVERGENCE = "settling_divergence"    # EP-specific: states don't converge
    SPECTRAL_INSTABILITY = "spectral_instability"  # σ(W) exceeds bound
    MEMORY_OOM = "memory_oom"
    TASK_INCOMPATIBILITY = "task_incompatibility"
    SLOW_CONVERGENCE = "slow_convergence"          # >3× baseline wall time
    NEGATIVE_TRANSFER = "negative_transfer"
    GOODNESS_COLLAPSE = "goodness_collapse"        # FF-specific: all goodness → 0
    SPIKE_SILENCING = "spike_silencing"             # STDP: all neurons go silent
```

**New file:** `bioplausible/analysis/failure_manifesto.py`

```python
class FailureManifestoGenerator:
    """Auto-generate reports/failure_manifesto.md from experiment DB."""
    def generate(self, output_path: str): ...
```

---

### 1.6 — Knowledgebase Enrichment

**New file:** `bioplausible/knowledge/metamodel.py`

```python
class KnowledgebaseMetamodel:
    """
    Phase 1: GP surrogate + symbolic regression.

    Capabilities:
    - predict(config) → (mean, std) for any metric
    - extract_rules() → human-readable symbolic formulas
    - top_k(task, k) → recommended configs with confidence bounds
    - explain(config) → natural-language justification
    """
    def fit(self, db_path: str): ...

    def extract_symbolic_rules(self) -> List[str]:
        """
        Use sympy curve-fitting or DEAP genetic programming.
        Example outputs:
        - "For EP on MNIST: accuracy ≈ 0.97 - 0.8·exp(-eq_steps/15)"
        - "For FF: depth > 4 AND lr < 3e-3 → accuracy > 0.92"
        - "Energy proxy decreases linearly with activation sparsity (R²=0.94)"
        """

    def compute_algorithm_similarity(self) -> np.ndarray:
        """Which algorithms behave similarly across tasks? (for clustering/taxonomy)."""
```

---

### 1.7 — Leaderboard and Reporting

**New file:** `bioplausible/leaderboard/`

```python
@dataclass
class LeaderboardEntry:
    algorithm: str
    optimizer: str
    task: str
    val_accuracy: float
    energy_proxy: float
    backward_flops: int          # 0 for bio-plausible methods
    requires_backward: bool
    param_count: int
    wall_time_s: float
    peak_memory_mb: float
    mean_acc: float              # across seeds
    std_acc: float
    config_hash: str
```

**Auto-generated outputs:**
- `reports/leaderboard.md` — GitHub-flavored markdown tables (sortable by accuracy, energy, memory).
- `reports/phase1_report.md` — arXiv-style technical report with all 5 signals, ablations, scaling curves, failure manifesto, and knowledgebase insights.

Report structure:
1. Abstract
2. Introduction: the case for a unified alternative-learning platform
3. The Bioplausible Framework (architecture, config system, Zoo)
4. Algorithms: taxonomy of 13 credit-assignment paradigms
5. Experimental Setup (controlled grid, compute envelope)
6. Results
   - Signal 1–5 tables and figures
   - Ablation heatmaps
   - Scaling curves
7. Failure Analysis
8. Knowledgebase Insights (symbolic rules extracted)
9. Discussion: what this means for neuromorphic computing
10. Conclusion + Phase 2 justification

---

### 1.8 — CI/CD and Tests

**File:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e ".[full]"
      - run: pytest tests/ -x -q --timeout=120
      - run: python examples/cross_domain_demo.py
```

**New file:** `tests/test_phase0.py`

```python
def test_config_roundtrip(): ...
def test_energy_proxy_logged(): ...
def test_knowledge_seed_fit(): ...
def test_forward_forward_trains(): ...
def test_pepita_trains(): ...
def test_mep_no_backward(): ...
def test_all_model_specs_have_metadata(): ...
```

---

## Milestones Timeline

| Week | Milestone | Key Deliverables |
|------|-----------|-----------------|
| W1 | Package + registry + config schema | `pip install -e .`, `RunConfig`, enriched `ModelSpec` |
| W2 | Runner + energy metrics + MEP integration | `run_from_config()`, `EnergyProfile` in every run |
| W2-3 | Forward-Forward + PEPITA implementation | Two new backward-free algorithms running on MNIST |
| W3 | DTP + Three-Factor + STDP (lightweight) | Five new families complete with unit tests |
| W3-4 | Graph/tabular domains + knowledgebase seed | `GraphTask`, `TabularTask`, `KnowledgebaseSeed` |
| **W4** | **Phase 0 complete** | **Cross-domain demo runs 5 paradigms** |
| W5-6 | Full experiment matrix execution (240 runs) | Results CSVs in `results/phase1/` |
| W7 | Signal 1-2: backward-free parity + energy frontier | Pareto plots, comparison tables |
| W8 | Signal 3-5: data efficiency, depth scaling, generality | Ablation DataFrames, scaling curves |
| W8-9 | Failure manifesto + knowledgebase enrichment | `failure_manifesto.md`, symbolic rules |
| W9-10 | Leaderboard + technical report | `leaderboard.md`, `phase1_report.md` |
| **W10** | **Phase 1 complete** | **arXiv draft + public repo + queryable KB** |

---

## The Competitive Moat

After Phase 1, Bioplausible occupies a position that **cannot be replicated** by:

| Competitor | Their Focus | What We Do Differently |
|---|---|---|
| PyTorch / JAX | Backprop infrastructure | We treat 12+ non-backprop paradigms as first-class citizens |
| snnTorch | Spiking networks only | We integrate spiking alongside 12 other paradigms on equal footing |
| Intel Lava | Loihi-specific | We're hardware-agnostic with `hardware_affinity` metadata |
| Brain.py / Nengo | Brain simulation | We produce competitive ML results, not just simulations |
| Hebbian libraries (various) | Single algorithm | We systematically compare Hebbian against 12 alternatives |
| No one | — | **Composable optimizer algebra** (gradient × update × constraint strategies from MEP) applied to any algorithm |

The knowledgebase — which learns *why* configurations succeed — creates a **compounding advantage** that grows with every experiment.
