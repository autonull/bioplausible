# REFACTOR.md — Bioplausible Codebase Reorganization Plan (No Backward Compat)

## Executive Summary

**Goal**: Single authoritative `README.md` as index; all components discoverable from it; minimal, non-redundant codebase organized by **algorithm families**. **No backward compatibility** — breaking changes accepted, version bump to 1.0.0.

**Critical Finding**: The "legacy" modules (`models/registry.py`, `models/factory.py`, `optimizers/learning_rules.py`, `optimizers/__init__.py`, `training/supervised.py`, `pipeline/`, `core.py`, `hybrid_optimizer.py`, `compat.py`) are **NOT dead code** — they are the current active APIs used by 100+ files across CLI, hyperopt, scientist, lightning, examples, tests, and UI. They must be **replaced by the new Zoo structure** with all imports migrated.

---

## 1. Documentation: Archive Everything, Expand README

### 1.1 Archive All Non-README Docs

```
MOVE → docs/archive/20260722/:
  - All /docs/*.md (44 files)
  - All root *.md except README.md, AGENTS.md, CONTRIBUTING.md, LICENSE, CHANGELOG.md
  - /mep/docs/* (5 files)
  - /mep/README*.md (2 files)
  - README0.md
  - bioplausible_ui/ (entire directory — per user request)
  - experiments/ one-off scripts (~21 files)
  - analysis_tools.py
  - config_legacy.py, config_loader.py, config_schema.py
  - cli.py (if audited as dead)
  - launch_studio.py, run_equitile_ui.py
  - gui.sh, run_ui.sh, clear_scientist.sh
```

**Keep at root**: `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`

**Keep at root (entry points)**: `lab.sh`, `run_benchmark.sh`, `run_benchmarks.sh`, `run_leaderboard.sh`, `run_scientist.sh`, `launch_leaderboard.py`, `smoke_test_all.py`, `test_formatting.py`

### 1.2 README.md as Complete Algorithm-Family Index

Every component = **one line + link to canonical source file**. Sections by algorithm family:

| Section | Canonical Source |
|---------|-----------------|
| Installation | `pyproject.toml` |
| Quick Start | `bioplausible/__init__.py` (CoreTrainer) |
| **EqProp Family** | `bioplausible/zoo/eqprop/` |
| &nbsp;&nbsp;LoopedMLP, StandardEqProp, DeepEP, ConvEqProp, ModernConvEqProp, EqPropDiffusion, TransformerEqProp, CausalTransformerEqProp, EqPropAttentionOnlyLM, FullEqPropLM, HybridEqPropLM, RecurrentEqPropLM, LoopedMLPForLM, MemoryEfficientLoopedMLP, NeuralCube, HomeostaticEqProp, TemporalResonanceEqProp, TernaryEqProp, SparseEquilibrium, MomentumEquilibrium, HolomorphicEP, FiniteNudgeEP, LazyEqProp, GraphEqProp | `zoo/eqprop/models.py`, `zoo/eqprop/propagators.py`, `zoo/eqprop/wrappers.py` |
| &nbsp;&nbsp;EqProp, HolomorphicEqProp, FiniteNudgeEqProp, LazyEqProp | `zoo/eqprop/propagators.py` |
| **Feedback Alignment Family** | `bioplausible/zoo/fa/` |
| &nbsp;&nbsp;StandardFA, DirectFeedbackAlignmentEqProp, AdaptiveFeedbackAlignment, EnergyGuidedFA, EnergyMinimizingFA, LayerwiseEquilibriumFA, EquilibriumAlignment | `zoo/fa/models.py`, `zoo/fa/propagators.py` |
| **Hebbian Family** | `bioplausible/zoo/hebbian/` |
| &nbsp;&nbsp;DeepHebbianChain, ThreeFactorHebbian, CHL | `zoo/hebbian/models.py`, `zoo/hebbian/propagators.py` |
| **Forward-Only Family (FF, PEPITA)** | `bioplausible/zoo/forward_only/` |
| &nbsp;&nbsp;ForwardForwardNet, PEPITA | `zoo/forward_only/models.py`, `zoo/forward_only/propagators.py` |
| **Target Propagation Family** | `bioplausible/zoo/target_prop/` |
| &nbsp;&nbsp;DifferenceTargetPropagation | `zoo/target_prop/models.py`, `zoo/target_prop/propagators.py` |
| **Spiking (STDP)** | `bioplausible/zoo/spiking/` |
| &nbsp;&nbsp;SpikingSTDP | `zoo/spiking/models.py`, `zoo/spiking/propagators.py` |
| **Predictive Coding (FabricPC)** | `bioplausible/zoo/predictive_coding/` |
| &nbsp;&nbsp;FabricPCGraphPCN, PredictiveCodingHybrid | `zoo/predictive_coding/models.py`, `zoo/predictive_coding/propagators.py` |
| **Backprop Baselines** | `bioplausible/zoo/backprop/` |
| &nbsp;&nbsp;BackpropMLP, BackpropTransformerLM | `zoo/backprop/models.py`, `zoo/backprop/propagators.py` |
| **MEP Optimizers** | `bioplausible/zoo/mep/` |
| &nbsp;&nbsp;smep, sdmep, local_ep, natural_ep, muon_backprop | `zoo/mep/presets.py` |
| &nbsp;&nbsp;Strategies: Muon, Dion, Spectral, EP gradients | `zoo/mep/strategies/` |
| &nbsp;&nbsp;Benchmarks | `zoo/mep/benchmarks/` |
| **EquiTile (Promoted to Top-Level)** | `bioplausible/equitile/` |
| &nbsp;&nbsp;EquiTile, ConvEquiTile, LMEquiTile, OptimizedLMEquiTile, RLEquiTile, RecurrentRLEquiTile, GraphEquiTile, TimeSeriesEquiTile, DynamicEquiTile, EnhancedEquiTile, EquiTileEP | `equitile/__init__.py` registers all |
| AutoScientist (Execution Engine) | `bioplausible/execution/` |
| AutoScientist (LLM Reasoner) | `bioplausible/autoscientist/` |
| Hyperparameter Optimization | `bioplausible/hyperopt/` |
| Validation Framework | `bioplausible/validation/` |
| Lightning Integration | `bioplausible/lightning_/` |
| Distributed / P2P | `bioplausible/p2p/` + `equitile/distributed.py` |
| Deployment / Export | `bioplausible/deployment.py` |
| Text Generation | `bioplausible/generation.py` |
| Scikit-learn Interface | `bioplausible/sklearn_interface.py` |
| Experiment Tracking | `bioplausible/tracking.py` |
| CLI | `bioplausible/cli/` |
| Analysis & Reporting | `bioplausible/analysis/` |
| Experiments (Reusable Infrastructure) | `bioplausible/experiments/utils.py`, `bioplausible/experiments/presets.py` |
| Examples | `examples/` (each with 1-line description) |

---

## 2. New Code Organization: Algorithm-Family Structure

### 2.1 Target Directory Layout

```
bioplausible/
├── __init__.py              # Public API: CoreTrainer, Registry, Zoo access
├── core/
│   ├── trainer.py           # CoreTrainer (unified training API)
│   ├── registry.py          # Single Registry for ALL components
│   └── energy.py            # Energy profiling (merged from top-level energy.py)
├── equitile/                # ← PROMOTED: top-level (was models/equitile/)
│   ├── __init__.py          # Registers ALL EquiTile variants
│   ├── core.py
│   ├── config.py
│   ├── builder.py
│   ├── dynamics.py
│   ├── enhanced.py
│   ├── language.py
│   ├── vision.py
│   ├── rl.py
│   ├── graph.py
│   ├── timeseries.py
│   ├── multigpu.py
│   ├── distributed.py
│   ├── async_execution.py
│   ├── deployment.py
│   ├── profiler.py
│   ├── research.py
│   ├── topology.py
│   ├── kernels.py
│   ├── benchmarks/
│   └── lm_demo/
├── zoo/                     # Algorithm families (each self-contained)
│   ├── __init__.py          # Exposes Registry, family discovery
│   ├── eqprop/
│   │   ├── __init__.py      # Registers models + propagators
│   │   ├── models.py        # All EqProp models (see mapping table)
│   │   ├── propagators.py   # EqProp, HolomorphicEqProp, FiniteNudge, LazyEqProp
│   │   ├── wrappers.py      # RecurrentWrapper, etc.
│   │   ├── base.py          # EqProp base class
│   │   └── configs.py
│   ├── fa/
│   │   ├── __init__.py
│   │   ├── models.py        # All FA models
│   │   ├── propagators.py   # FA, DirectFA, AdaptiveFA, StochasticFA, ContrastiveFA
│   │   └── configs.py
│   ├── hebbian/
│   │   ├── __init__.py
│   │   ├── models.py        # DeepHebbianChain, ThreeFactorHebbian
│   │   ├── propagators.py   # CHL propagator
│   │   └── configs.py
│   ├── forward_only/
│   │   ├── __init__.py
│   │   ├── models.py        # ForwardForwardNet, PEPITA
│   │   ├── propagators.py   # FF, PEPITA propagators
│   │   └── configs.py
│   ├── target_prop/
│   │   ├── __init__.py
│   │   ├── models.py        # DifferenceTargetProp
│   │   ├── propagators.py   # Target propagation
│   │   └── configs.py
│   ├── spiking/
│   │   ├── __init__.py
│   │   ├── models.py        # SpikingSTDP
│   │   ├── propagators.py   # STDP propagator
│   │   └── configs.py
│   ├── predictive_coding/
│   │   ├── __init__.py
│   │   ├── models.py        # FabricPCGraphPCN, PredictiveCodingHybrid
│   │   ├── propagators.py   # PCN propagator
│   │   └── configs.py
│   ├── backprop/
│   │   ├── __init__.py
│   │   ├── models.py        # BackpropMLP, BackpropTransformerLM
│   │   ├── propagators.py   # BackpropPropagator (standard autograd)
│   │   └── configs.py
│   └── mep/                 # ← MOVED from /mep/mep/
│       ├── __init__.py      # Registers presets as propagators, strategies as optimizers
│       ├── optimizers/
│       │   ├── composite.py
│       │   ├── ep_optimizer.py
│       │   ├── settling.py
│       │   ├── energy.py
│       │   ├── ewc.py
│       │   ├── o1_memory.py
│       │   └── o1_memory_v2.py
│       ├── strategies/
│       │   ├── gradient.py      # EPGradient, LocalEPGradient, NaturalGradient
│       │   ├── update.py        # PlainUpdate, MuonUpdate, DionUpdate, FisherUpdate
│       │   ├── constraint.py    # NoConstraint, SpectralConstraint
│       │   └── feedback.py      # NoFeedback, ErrorFeedback
│       ├── presets.py           # smep(), sdmep(), local_ep(), natural_ep(), muon_backprop()
│       ├── monitor.py
│       ├── inspector.py
│       ├── benchmarks/
│       │   ├── baselines.py
│       │   ├── compare.py
│       │   ├── metrics.py
│       │   ├── runner.py
│       │   ├── config/
│       │   ├── niche_benchmarks.py
│       │   ├── continual_learning.py
│       │   ├── ewc_baseline.py
│       │   ├── tuned_compare.py
│       │   └── visualization.py
│       └── cuda/
│           ├── __init__.py
│           └── kernels.py
├── execution/               # ← RENAMED from scientist/
│   ├── engine.py            # ← was core.py (ExecutionEngine)
│   ├── task.py              # ExperimentTask
│   ├── strategy.py          # ExecutionStrategy
│   ├── state.py             # ExperimentState
│   ├── dashboard.py         # Dashboard
│   ├── monitoring.py
│   ├── resources.py
│   ├── failure_tracker.py
│   ├── promotion.py
│   ├── robustness.py
│   ├── safety.py
│   ├── interpretability.py
│   ├── experiment_checks.py
│   ├── decisions.py
│   ├── curriculum.py
│   ├── checkpoint_manager.py
│   ├── archiver.py
│   ├── algorithm_constraints.py
│   ├── evolve_evaluator.py
│   └── training_dynamics.py
├── autoscientist/           # LLM reasoner (unchanged)
│   ├── bridge.py
│   ├── campaign.py
│   ├── proposer.py
│   └── reasoner.py
├── hyperopt/                # Optuna integration
├── validation/
│   ├── core.py              # Verifier
│   ├── notebook.py          # VerificationNotebook
│   └── tracks/              # Consolidated to 9 files + TrackRegistry
│       ├── __init__.py
│       ├── core_tracks.py
│       ├── scaling_tracks.py
│       ├── research_tracks.py
│       ├── signal_tracks.py
│       ├── tradeoff_tracks.py
│       ├── hardware_tracks.py
│       ├── application_tracks.py
│       ├── architecture_comparison.py
│       ├── negative_results.py
│       └── nebc_tracks.py
├── lightning_/              # PyTorch Lightning integration
├── p2p/                     # P2P coordinator
├── deployment.py            # ONNX/TorchScript export, inference server (merged from export.py)
├── cli/                     # CLI (run.py, lab.py, rank.py)
├── datasets.py              # Data loading
├── domains/                 # Domain/task definitions
├── graph/                   # FabricPC graph API (implementation detail)
├── analysis/                # Analysis infrastructure
│   ├── __init__.py
│   ├── results.py
│   ├── scaling.py
│   ├── ablation.py
│   ├── dynamics.py
│   ├── failure_manifesto.py
│   └── reporting.py
├── experiments/             # Reusable experiment infrastructure
│   ├── __init__.py
│   ├── utils.py             # ExperimentRunner, HyperparameterSearch, quick_comparison, benchmark_model
│   └── presets.py           # ResearchPreset, category-based preset discovery (uses Registry)
├── acceleration/            # Hardware acceleration
│   ├── __init__.py
│   ├── triton_kernels.py    # (merged from kernel.py, triton_kernel.py)
│   ├── compile.py
│   ├── backends.py
│   └── kernels.py
├── config/                  # OmegaConf/Pydantic config schemas
│   ├── __init__.py          # (merged from config_loader.py)
│   ├── schema.py            # (merged from config_schema.py)
│   └── ...
├── evaluation/              # MetricSuite, EvaluatorBase, BenchmarkRegistry
├── knowledge/               # KnowledgeBase for experiment metadata
├── leaderboard/             # LeaderboardGenerator
├── utils.py
├── visualization.py
├── visualization_tools.py
├── statistics.py
├── generation.py            # Autoregressive text generation
├── sklearn_interface.py     # Scikit-learn compatible wrapper
├── tracking.py              # WandB experiment tracking wrapper
└── runner.py                # DEPRECATED — thin wrapper, merge into CoreTrainer.run_from_config()
```

### 2.2 What Gets DELETED (Not Moved)

| Path | Reason |
|------|--------|
| `/mep/` (root) | Own pyproject.toml; contents moved to `zoo/mep/` |
| `bioplausible/compat.py` | 10K lines of backward compat — DELETE |
| `bioplausible/models/registry.py` | Legacy registry — REPLACED by `core/registry.py` |
| `bioplausible/models/factory.py` | Legacy factory — REPLACED by Zoo |
| `bioplausible/optimizers/learning_rules.py` | Canonical impls moved to `zoo/*/propagators.py` |
| `bioplausible/optimizers/base.py` | Replaced by `zoo/mep/optimizers/composite.py` |
| `bioplausible/optimizers/__init__.py` | Thin wrapper — DELETE |
| `bioplausible/pipeline/` | Superseded by CoreTrainer — DELETE |
| `bioplausible/training/supervised.py` | 36K lines, superseded by CoreTrainer — DELETE |
| `bioplausible/training/base.py` | DELETE |
| `bioplausible/training/rl.py` | KEEP (RL-specific, different paradigm) |
| `bioplausible/models/tile_eq.py` | Redundant with equitile/ — DELETE (after audit) |
| `bioplausible/hybrid_optimizer.py` | Prototype — DELETE |
| `bioplausible/core.py` | Thin alias for SupervisedTrainer — DELETE |
| `bioplausible/zoo/models/registered_models.py` | Stubs — DELETE (replaced by family `__init__.py`) |
| `bioplausible/zoo/propagators/registered_propagators.py` | Wrappers — DELETE |
| `bioplausible/zoo/optimizers/registered_optimizers.py` | Wrappers — DELETE |
| `bioplausible/zoo/sparsity/registered_sparsity.py` | DELETE (move to family dirs or `zoo/sparsity.py`) |
| `bioplausible/scientist/` (entire dir) | Renamed to `execution/` — DELETE old |
| 8 redundant validation track files | Consolidated into 9 core files — DELETE |
| `bioplausible/experiments/` one-off scripts (~21) | Archive to `docs/archive/20260722/experiments/` |
| `bioplausible/analysis_tools.py` | Overlaps with `analysis/` — archive after extracting unique code |
| `bioplausible/cli.py` | Audit — if dead, archive; if unique commands, merge into `cli/` |
| `bioplausible/launch_studio.py`, `run_equitile_ui.py` | UI-related — archive |
| `gui.sh`, `run_ui.sh`, `clear_scientist.sh` | UI/maintenance — archive |

---

## 3. Registry Pattern: Per-Family Registration

### 3.1 Single Registry (`core/registry.py`)

Unchanged — `Registry` with `@register_model`, `@register_propagator`, `@register_optimizer`, `@register_sparsity`, `@register_track`.

### 3.2 Each Family Registers in Its `__init__.py`

```python
# bioplausible/zoo/eqprop/__init__.py
from bioplausible.core.registry import (
    register_model, register_propagator, Domain, LocalityLevel, ComputeProfile
)

# Models (imported from models.py)
from .models import (
    LoopedMLP, StandardEqProp, DeepEP, ConvEqProp, ModernConvEqProp,
    EqPropDiffusion, TransformerEqProp, CausalTransformerEqProp,
    EqPropAttentionOnlyLM, FullEqPropLM, HybridEqPropLM,
    RecurrentEqPropLM, LoopedMLPForLM, MemoryEfficientLoopedMLP,
    NeuralCube, HomeostaticEqProp, TemporalResonanceEqProp,
    TernaryEqProp, SparseEquilibrium, MomentumEquilibrium,
    HolomorphicEP, FiniteNudgeEP, LazyEqProp, GraphEqProp,
)

# Propagators (imported from propagators.py)
from .propagators import (
    EqProp, HolomorphicEqProp, FiniteNudgeEqProp, LazyEqProp,
)

# Wrappers (imported from wrappers.py)
from .wrappers import RecurrentWrapper

# Register models
register_model(LoopedMLP, name="LoopedMLP",
    domains=[Domain.VISION, Domain.RL],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    ...)

# ... register each model with rich metadata

# Register propagators
register_propagator(EqProp, name="EqProp",
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    ...)

# ... etc
```

**No stubs, no thin wrappers** — real classes decorated directly.

---

## 4. EquiTile: Top-Level Package

**Rationale**: 20+ modules, benchmarks, demos, research tools — it's a sub-framework.

**Location**: `bioplausible/equitile/`

**Registration in `equitile/__init__.py`**:
```python
from bioplausible.core.registry import register_model
from .core import EquiTile, EquiTileEP
from .dynamics import DynamicEquiTile
from .enhanced import EnhancedEquiTile
from .vision import ConvEquiTile
from .language import LMEquiTile, OptimizedLMEquiTile
from .rl import RLEquiTile, RecurrentRLEquiTile
from .graph import GraphEquiTile
from .timeseries import TimeSeriesEquiTile

register_model(EquiTile, name="EquiTile", domains=[Domain.VISION, Domain.RL, Domain.LM], ...)
register_model(ConvEquiTile, name="ConvEquiTile", domains=[Domain.VISION], ...)
# ... all variants
```

---

## 5. MEP: Algorithm Family in Zoo

**From**: `/mep/mep/` (top-level package with own pyproject.toml)

**To**: `bioplausible/zoo/mep/`

**Registration in `zoo/mep/__init__.py`**:
```python
from bioplausible.core.registry import (
    register_propagator, register_optimizer, Domain, LocalityLevel
)
from .presets import smep, smep_fast, sdmep, local_ep, natural_ep, muon_backprop
from .strategies.update import PlainUpdate, MuonUpdate, DionUpdate, FisherUpdate
from .strategies.gradient import EPGradient, LocalEPGradient, NaturalGradient
from .strategies.constraint import NoConstraint, SpectralConstraint
from .strategies.feedback import NoFeedback, ErrorFeedback

# Composite presets = Propagators (credit assignment + update combined)
register_propagator(smep, name="smep", locality=LocalityLevel.EQUILIBRIUM, ...)
register_propagator(smep_fast, name="smep_fast", ...)
register_propagator(sdmep, name="sdmep", ...)
register_propagator(local_ep, name="local_ep", ...)
register_propagator(natural_ep, name="natural_ep", ...)
register_propagator(muon_backprop, name="muon_backprop", requires_backward=True, ...)

# Pure update strategies = Optimizers
register_optimizer(MuonUpdate, name="muon", ...)
register_optimizer(DionUpdate, name="dion", ...)
register_optimizer(PlainUpdate, name="plain", ...)
```

**Update `pyproject.toml`**: Remove `mep` package, add `bioplausible.zoo.mep` and `bioplausible.equitile`.

---

## 6. Execution Engine (was Scientist)

**Rename**: `bioplausible/scientist/` → `bioplausible/execution/`

**Class Renames**:
- `Scientist` → `ExecutionEngine`
- `AutoScientist` (alias) → **REMOVED**
- All other classes keep names (`ExperimentTask`, `ExecutionStrategy`, etc.)

**All imports updated** to use Zoo registry for model/propagator/optimizer discovery.

---

## 7. Validation Tracks: Consolidate to 9 Files

**Keep** (merge redundant content):
1. `core_tracks.py` — Smoke, unit, integration
2. `scaling_tracks.py` — Depth, width, data scaling
3. `research_tracks.py` — Novel algorithm evaluation
4. `signal_tracks.py` — Dynamics, gradient analysis
5. `tradeoff_tracks.py` — Perf vs compute (honest tradeoff)
6. `hardware_tracks.py` — GPU/CPU/neuromorphic
7. `application_tracks.py` — Vision, LM, RL, tabular
8. `architecture_comparison.py` — Model-to-model
9. `negative_results.py` — Failed approaches
10. `nebc_tracks.py` — NEBC assessment

**Delete** (8 files — merge content above or discard):
- `advanced_tracks.py`, `analysis_tracks.py`, `engine_validation_tracks.py`
- `enhanced_validation_tracks.py`, `framework_validation.py`
- `new_tracks.py`, `rapid_validation.py`, `special_tracks.py`

**TrackRegistry** in `validation/tracks/__init__.py` with `@register_track`.

---

## 8. Complete Model File → Family Mapping

### 8.1 Model Files

| File | Family Target | Notes |
|------|---------------|-------|
| `looped_mlp.py` | `zoo/eqprop/models.py` | LoopedMLP, BackpropMLP |
| `standard_eqprop.py` | `zoo/eqprop/models.py` | StandardEqProp |
| `conv_eqprop.py` | `zoo/eqprop/models.py` | ConvEqProp |
| `deep_ep.py` | `zoo/eqprop/models.py` | DeepEP |
| `memory_efficient.py` | `zoo/eqprop/models.py` | MemoryEfficientLoopedMLP |
| `transformer_eqprop.py` | `zoo/eqprop/models.py` | TransformerEqProp |
| `causal_transformer_eqprop.py` | `zoo/eqprop/models.py` | CausalTransformerEqProp |
| `eqprop_diffusion.py` | `zoo/eqprop/models.py` | EqPropDiffusion |
| `holomorphic_ep.py` | `zoo/eqprop/models.py` | HolomorphicEP |
| `finite_nudge_ep.py` | `zoo/eqprop/models.py` | FiniteNudgeEP |
| `lazy_eqprop.py` | `zoo/eqprop/models.py` | LazyEqProp |
| `neural_cube.py` | `zoo/eqprop/models.py` | NeuralCube |
| `temporal_resonance.py` | `zoo/eqprop/models.py` | TemporalResonanceEqProp |
| `ternary.py` | `zoo/eqprop/models.py` | TernaryEqProp |
| `sparse_eq.py` | `zoo/eqprop/models.py` | SparseEquilibrium |
| `mom_eq.py` | `zoo/eqprop/models.py` | MomentumEquilibrium |
| `homeostatic.py` | `zoo/eqprop/models.py` | HomeostaticEqProp |
| `modern_conv_eqprop.py` | `zoo/eqprop/models.py` | ModernConvEqProp |
| `eqprop_lm_variants.py` | `zoo/eqprop/models.py` | FullEqPropLM, EqPropAttentionOnlyLM, RecurrentEqPropLM, HybridEqPropLM, LoopedMLPForLM |
| `graph_eqprop.py` | `zoo/eqprop/models.py` | GraphEqProp — **add to README** |
| `feedback_alignment.py` | `zoo/fa/models.py` | AdaptiveFeedbackAlignment |
| `dfa_eqprop.py` | `zoo/fa/models.py` | DirectFeedbackAlignmentEqProp |
| `simple_fa.py` | `zoo/fa/models.py` | StandardFA — **add to README** |
| `eg_fa.py` | `zoo/fa/models.py` | EnergyGuidedFA |
| `em_fa.py` | `zoo/fa/models.py` | EnergyMinimizingFA |
| `leq_fa.py` | `zoo/fa/models.py` | LayerwiseEquilibriumFA |
| `eq_align.py` | `zoo/fa/models.py` | EquilibriumAlignment |
| `hebbian_chain.py` | `zoo/hebbian/models.py` | DeepHebbianChain |
| `three_factor.py` | `zoo/hebbian/models.py` | ThreeFactorHebbian |
| `chl.py` | `zoo/hebbian/models.py` (or propagators) | ContrastiveHebbianLearning — straddles boundary |
| `forward_forward.py` | `zoo/forward_only/models.py` | Forward-Forward |
| `pepita.py` | `zoo/forward_only/models.py` | PEPITA |
| `spiking_stdp.py` | `zoo/spiking/models.py` | SpikingSTDP |
| `target_prop.py` | `zoo/target_prop/models.py` | DifferenceTargetPropagation |
| `fabricpc_graph_pcn.py` | `zoo/predictive_coding/models.py` | FabricPCGraphPCN |
| `backprop_transformer_lm.py` | `zoo/backprop/models.py` | BackpropTransformerLM |
| `pc_hybrid.py` | `zoo/predictive_coding/models.py` | PredictiveCodingHybrid — **add to README** |
| `custom_stack.py` | `zoo/backprop/models.py` or `utils.py` | Generic layer builder — utility |

### 8.2 Base Classes and Support Files

| File | Action |
|------|--------|
| `eqprop_base.py` | **Move** to `zoo/eqprop/` as `base.py` — EqProp model base class |
| `eqprop_wrappers.py` | **Move** to `zoo/eqprop/` as `wrappers.py` — RecurrentWrapper, etc. |
| `nebc_base.py` | **Archive** or **move** to `zoo/nebc_base.py` — abstract base, only 2-3 usages |
| `base.py` | **Keep** at `zoo/base.py` — BioModel base used by StandardFA, PredictiveCodingHybrid, etc. |
| `tile_eq.py` | **Delete** — superseded by equitile/ |
| `benchmark.py` | **Archive** — standalone benchmark script |
| `utils.py` | **Move** to `zoo/utils.py` or merge into `bioplausible/utils.py` |
| `triton_kernel.py` | **Move** to `acceleration/triton_kernels.py` |

### 8.3 Propagator Files (from `optimizers/`)

| File | Family Target |
|------|---------------|
| `optimizers/learning_rules.py` | Split into `zoo/*/propagators.py` per family |
| `optimizers/base.py` | Delete (replaced by new structure) |
| `optimizers/__init__.py` | Delete |

---

## 9. Top-Level File Disposition (Complete Map)

| File | Lines | Action |
|------|-------|--------|
| `energy.py` | 93 | **Merge** into `core/energy.py` |
| `export.py` | 123 | **Merge** into `deployment.py` |
| `generation.py` | 153 | **Keep** at `bioplausible/generation.py` |
| `kernel.py` | 997 | **Move** to `acceleration/kernels.py` (merge with triton_kernel.py) |
| `runner.py` | 136 | **Merge** into `core/trainer.py` as `CoreTrainer.run_from_config()` |
| `tracking.py` | 140 | **Keep** at `bioplausible/tracking.py` |
| `sklearn_interface.py` | 287 | **Keep** at `bioplausible/sklearn_interface.py` |
| `analysis_tools.py` | 470 | **Deprecate** — audit, extract unique code to `analysis/`, then archive |
| `cli.py` | 135 | **Audit** — if dead, archive; else merge unique commands into `cli/` |
| `config_legacy.py` | 190 | **Keep** but mark `# DEPRECATED` at top, do NOT import from new code |
| `config_loader.py` | 52 | **Merge** into `config/__init__.py` |
| `config_schema.py` | 62 | **Merge** into `config/schema.py` |
| `compat.py` | 391 | **Delete** |
| `core.py` | 9 | **Delete** |
| `hybrid_optimizer.py` | 327 | **Delete** |

---

## 10. Packages That Stay As-Is (Already Well-Structured)

| Package | Purpose |
|---------|---------|
| `bioplausible/config/` | OmegaConf/Pydantic config schemas |
| `bioplausible/data/` | Data loading (vision, LM, curricula) |
| `bioplausible/domains/` | DomainTask definitions (Vision, LM, RL, Graph, etc.) |
| `bioplausible/acceleration/` | Hardware acceleration (triton, compile, backends) |
| `bioplausible/evaluation/` | MetricSuite, EvaluatorBase, BenchmarkRegistry |
| `bioplausible/knowledge/` | KnowledgeBase for experiment metadata |
| `bioplausible/leaderboard/` | LeaderboardGenerator |
| `bioplausible/graph/` | FabricPC graph API (implementation detail) |
| `bioplausible/lightning_/` | PyTorch Lightning integration |
| `bioplausible/p2p/` | P2P coordinator system |
| `bioplausible/cli/` | CLI entry points |
| `bioplausible/hyperopt/` | Optuna integration |
| `bioplausible/autoscientist/` | LLM reasoner |
| `bioplausible/validation/` | Validation framework |

---

## 11. Import Migration Map (Complete)

| Old Import | New Import |
|------------|------------|
| `from bioplausible.models.registry import get_model_spec, MODEL_REGISTRY, list_model_names` | `from bioplausible.core.registry import Registry` → `Registry.get("model", name)` / `Registry.list("model")` |
| `from bioplausible.models.factory import create_model, load_weights` | `from bioplausible.zoo import create_model` (new helper) |
| `from bioplausible.optimizers import create_optimizer, list_optimizers, FeedbackAlignment, EqProp, smep` | `from bioplausible.core.registry import Registry` → `Registry.get("propagator", "FeedbackAlignment")` / `Registry.get("optimizer", "smep")` |
| `from bioplausible.optimizers.learning_rules import FeedbackAlignment, EqProp, ...` | `from bioplausible.zoo.eqprop.propagators import EqProp` / `from bioplausible.zoo.fa.propagators import FeedbackAlignment` |
| `from mep import smep, sdmep, local_ep, natural_ep, muon_backprop` | `from bioplausible.zoo.mep.presets import smep, sdmep, local_ep, natural_ep, muon_backprop` |
| `from mep.optimizers import CompositeOptimizer, EPGradient, MuonUpdate, ...` | `from bioplausible.zoo.mep.optimizers import CompositeOptimizer` / `from bioplausible.zoo.mep.strategies import EPGradient, MuonUpdate, ...` |
| `from bioplausible.scientist import Scientist, AutoScientist, ExperimentTask` | `from bioplausible.execution import ExecutionEngine, ExperimentTask` |
| `from bioplausible.scientist.core import Scientist` | `from bioplausible.execution.engine import ExecutionEngine` |
| `from bioplausible.scientist.strategy import ScientistStrategy` | `from bioplausible.execution.strategy import ExecutionStrategy` |
| `from bioplausible.scientist.state import ExperimentState` | `from bioplausible.execution.state import ExperimentState` |
| `from bioplausible.scientist.task import ExperimentTask` | `from bioplausible.execution.task import ExperimentTask` |
| `from bioplausible.training.supervised import SupervisedTrainer` | `from bioplausible.core.trainer import CoreTrainer` |
| `from bioplausible.core import EqPropTrainer` | **DELETE** — was alias for SupervisedTrainer |
| `from bioplausible.hybrid_optimizer import HybridEqPropOptimizer` | **DELETE** — prototype |
| `from bioplausible.pipeline.config import TrainingConfig` | `from bioplausible.core.trainer import TrainerConfig` |
| `from bioplausible.pipeline.session import TrainingSession, SessionState` | `from bioplausible.core.trainer import CoreTrainer` (session pattern removed) |
| `from bioplausible.pipeline.events import ...` | **DELETE** — event system removed |
| `from bioplausible.pipeline.results import ResultsManager` | **DELETE** — results in CoreTrainer history |
| `from bioplausible.models.equitile import EquiTile, ConvEquiTile, ...` | `from bioplausible.equitile import EquiTile, ConvEquiTile, ...` |
| `from bioplausible.validation.tracks import core_tracks, scaling_tracks, ...` | `from bioplausible.validation.tracks import TrackRegistry` |
| `from bioplausible.models.*` (any model) | `from bioplausible.zoo.{family}.models import ModelName` |
| `from bioplausible.experiments.utils import ...` | Update to use `Registry.get()` for model/optimizer discovery |

---

## 12. Execution Phases (Corrected Order)

### Phase 1: Documentation Archive & README (1 day)
1. Create `docs/archive/20260722/`
2. Move all files per §1.1
3. Write complete `README.md` per §1.2
4. Verify all links resolve

### Phase 2: Create New Structure & Move Contents (2 days)
1. Create algorithm-family dirs: `zoo/eqprop/`, `zoo/fa/`, `zoo/hebbian/`, `zoo/forward_only/`, `zoo/target_prop/`, `zoo/spiking/`, `zoo/predictive_coding/`, `zoo/backprop/`
2. Create `equitile/` at top level
3. Create `execution/` dir
4. Move `/mep/mep/` → `bioplausible/zoo/mep/` (including benchmarks/, cuda/)
5. Move `bioplausible/models/equitile/` → `bioplausible/equitile/`
6. Move model implementations from `models/*.py` into appropriate family `models.py` (per §8 mapping)
7. Move propagator implementations from `optimizers/learning_rules.py` into family `propagators.py`
8. Move `scientist/` files to `execution/` with renamed files (`core.py` → `engine.py`, etc.)
9. Consolidate validation tracks to 9 files + `TrackRegistry`
10. Move top-level files per §9 disposition
11. Update `pyproject.toml` packages list

### Phase 3: Register Everything (1 day)
1. Each family `__init__.py` registers models + propagators with rich metadata
2. `equitile/__init__.py` registers all EquiTile variants
3. `zoo/mep/__init__.py` registers presets as propagators, strategies as optimizers
4. `validation/tracks/__init__.py` creates TrackRegistry, registers tracks
5. Update `bioplausible/__init__.py` with clean public API

### Phase 4: Delete Deprecated Files & Directories (0.5 day)
**Must not run until all content has been extracted to new locations.**
1. Delete `bioplausible/compat.py`
2. Delete `bioplausible/models/` (entire dir)
3. Delete `bioplausible/optimizers/learning_rules.py`, `optimizers/base.py`, `optimizers/__init__.py`
4. Delete `bioplausible/pipeline/`
5. Delete `bioplausible/training/supervised.py`, `training/base.py`
6. Delete `bioplausible/core.py`, `bioplausible/hybrid_optimizer.py`
7. Delete `bioplausible/models/tile_eq.py`
8. Delete `bioplausible/zoo/models/registered_models.py`, `zoo/propagators/registered_propagators.py`, `zoo/optimizers/registered_optimizers.py`, `zoo/sparsity/registered_sparsity.py`
9. Delete `bioplausible/scientist/` (entire dir)
10. Delete 8 redundant validation track files
11. Delete `/mep/` root (keep `/mep/mep/` contents already moved)
12. Archive `bioplausible_ui/` → `docs/archive/20260722/bioplausible_ui/`
13. Archive experiments one-off scripts
14. Archive `analysis_tools.py`, `config_legacy.py`, `cli.py` (if audited dead), `launch_studio.py`, `run_equitile_ui.py`, `gui.sh`, `run_ui.sh`, `clear_scientist.sh`

### Phase 5: Update All Consumers (1.5 days)
1. `cli/run.py`, `cli/lab.py` → Zoo registry + CoreTrainer
2. `hyperopt/` → Zoo registry for model/optimizer/propagator lookup
3. `execution/` (was scientist) → Zoo registry + CoreTrainer
4. `lightning_/` → Zoo registry for optimizers
5. `autoscientist/bridge.py` → Zoo registry for component discovery
6. `core/trainer.py` → Zoo registry for model/optimizer lookup
7. `examples/` → Updated imports
8. `tests/` → Updated imports (fix all test files)
9. `experiments/utils.py`, `experiments/presets.py` → Use `Registry.get()` instead of hardcoded strings
10. `analysis/` → Update imports from old paths
11. `runner.py` → Merge into `CoreTrainer.run_from_config()`, then delete

### Phase 6: Format, Lint, Test (0.5 day)
1. `isort . && black .`
2. `flake8`
3. `pytest tests/` — fix failures
4. Verify `README.md` completeness

---

## 13. UI Code: Archive Entirely

```
MOVE → docs/archive/20260722/bioplausible_ui/:
  /home/me/bioplausible/bioplausible_ui/ (entire directory)
```

**Rationale**: Not complete, not used, separate package. Archive as-is for potential future resurrection.

---

## 14. Risk Assessment

| Risk | Mitigation |
|------|------------|
| External user imports break | **Accepted** — v1.0.0, no compat layer |
| Tests fail en masse | Phase 6 fixes; most are import updates |
| AutoScientist campaigns break | Safe — campaigns use DB, not imports |
| Forgotten import in obscure file | `grep -r` patterns in Phase 5 checklist |
| `bioplausible_ui` needs updates | Archived — not a concern |

---

## 15. Success Criteria

1. **Single docs entry**: `README.md` complete, only file new users need
2. **Single registry**: `Registry` in `core/registry.py` = source of truth
3. **Algorithm-family org**: `zoo/eqprop/`, `zoo/fa/`, `zoo/mep/`, `equitile/` — clear ownership
4. **No redundancy**: Each algorithm implemented once, registered once, documented once
5. **Clear naming**: Propagator=credit assignment, Optimizer=parameter update, Engine=execution, AutoScientist=LLM reasoner
6. **All tests pass**: No regressions
7. **~60 docs archived, ~44 Python files deleted, 2 registries → 1, 1 top-level package eliminated, UI archived**
8. **No orphan imports**: `grep -r "from bioplausible.models"` and `from bioplausible.optimizers` return 0 hits after Phase 5
9. **All 3 undocumented models** (GraphEqProp, PredictiveCodingHybrid, StandardFA) are in README
10. **`experiments/presets.py`** resolves models through Zoo Registry, not hardcoded strings
11. **All packages in §10** have explicit `__init__.py` entries for their exports
12. **`plot_results.py`, `generate_report.sh` etc.** verified to work after path migrations
13. **`mep/mep/benchmarks/`** runs correctly from new `zoo/mep/benchmarks/` location

---

## 16. File Count Impact (Corrected)

| Category | Before | After | Delta |
|----------|--------|-------|-------|
| Root `.md` files | 9 | 6 | -3 |
| `/docs/` `.md` files | 44 | 0 (archived) | -44 |
| `/mep/` (root) | 1 package | 0 | -1 |
| `bioplausible/compat.py` | 391 lines | 0 | -391 |
| `bioplausible/models/` (entire dir) | 51 files | 0 | -51 |
| `bioplausible/optimizers/learning_rules.py` | 814 lines | 0 | -814 |
| `bioplausible/optimizers/base.py` + `__init__.py` | 2 files | 0 | -2 |
| `bioplausible/pipeline/` | 4 files | 0 | -4 |
| `bioplausible/training/supervised.py` + `base.py` | 958 lines | 0 | -958 |
| `bioplausible/core.py` + `hybrid_optimizer.py` | 336 lines | 0 | -336 |
| `bioplausible/models/tile_eq.py` | 2705 lines | 0 | -2705 |
| `bioplausible/zoo/*/registered_*.py` | 4 files | 0 | -4 |
| `bioplausible/scientist/` | 1 dir | 0 (→ `execution/`) | 0 |
| `bioplausible/experiments/` one-off scripts | ~21 files | 0 (archived) | -21 |
| `bioplausible/analysis_tools.py` | 470 lines | 0 (archived) | -470 |
| Validation track files | 20 | 9 | -11 |
| `bioplausible_ui/` | 1 dir | 0 (archived) | -1 |

**Net**: ~44 files deleted, ~6K lines of dead/deprecated code removed, ~3K lines of one-off experiments archived. Cleaner hierarchy with no functional loss.

---