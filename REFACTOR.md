# REFACTOR.md — Bioplausible Codebase Reorganization Plan (No Backward Compat)

## Executive Summary

**Goal**: Single authoritative `README.md` as index; all components discoverable from it; minimal, non-redundant codebase organized by **algorithm families**. **No backward compatibility** — breaking changes accepted, version bump to 1.0.0.

**Critical Finding**: The "legacy" modules (`models/registry.py`, `models/factory.py`, `optimizers/learning_rules.py`, `optimizers/__init__.py`, `training/supervised.py`, `pipeline/`, `core.py`, `hybrid_optimizer.py`, `compat.py`) are **NOT dead code** — they are the current active APIs used by 100+ files across CLI, hyperopt, scientist, lightning, examples, tests, and UI. They must be **replaced by the new Zoo structure** with all imports migrated.

---

## 1. Documentation: Archive Everything, Expand README

### 1.1 Archive All Non-README Docs

```
MOVE → docs/archive/20260722/:
  - All /docs/*.md (43 files)
  - All root *.md except README.md, AGENTS.md, CONTRIBUTING.md, LICENSE, CHANGELOG.md
  - /mep/docs/* (5 files)
  - /mep/README*.md (2 files)
  - README0.md
```

**Keep at root**: `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`

### 1.2 README.md as Complete Algorithm-Family Index

Every component = **one line + link to canonical source file**. Sections by algorithm family:

| Section | Canonical Source |
|---------|-----------------|
| Installation | `pyproject.toml` |
| Quick Start | `bioplausible/__init__.py` (CoreTrainer) |
| **EqProp Family** | `bioplausible/zoo/eqprop/` |
| &nbsp;&nbsp;LoopedMLP, StandardEqProp, DeepEP, ConvEqProp, etc. | `zoo/eqprop/models.py`, `zoo/eqprop/propagators.py` |
| **Feedback Alignment Family** | `bioplausible/zoo/fa/` |
| &nbsp;&nbsp;FA, DirectFA, AdaptiveFA, StochasticFA, ContrastiveFA | `zoo/fa/models.py`, `zoo/fa/propagators.py` |
| **Hebbian Family** | `bioplausible/zoo/hebbian/` |
| &nbsp;&nbsp;DeepHebbianChain, ThreeFactorHebbian, CHL | `zoo/hebbian/models.py`, `zoo/hebbian/propagators.py` |
| **Forward-Only Family (FF, PEPITA)** | `bioplausible/zoo/forward_only/` |
| **Target Propagation Family** | `bioplausible/zoo/target_prop/` |
| **Spiking (STDP)** | `bioplausible/zoo/spiking/` |
| **Predictive Coding (FabricPC)** | `bioplausible/zoo/predictive_coding/` |
| **Backprop Baselines** | `bioplausible/zoo/backprop/` |
| **MEP Optimizers** | `bioplausible/zoo/mep/` |
| &nbsp;&nbsp;smep, sdmep, local_ep, natural_ep, muon_backprop | `zoo/mep/presets.py` |
| &nbsp;&nbsp;Strategies: Muon, Dion, Spectral, EP gradients | `zoo/mep/strategies/` |
| **EquiTile (Promoted to Top-Level)** | `bioplausible/equitile/` |
| &nbsp;&nbsp;EquiTile, ConvEquiTile, LMEquiTile, RLEquiTile, etc. | `equitile/__init__.py` registers all |
| AutoScientist (Execution Engine) | `bioplausible/execution/` |
| AutoScientist (LLM Reasoner) | `bioplausible/autoscientist/` |
| Hyperparameter Optimization | `bioplausible/hyperopt/` |
| Validation Framework | `bioplausible/validation/` |
| Lightning Integration | `bioplausible/lightning_/` |
| Distributed / P2P | `bioplausible/p2p/` + `equitile/distributed.py` |
| Deployment / Export | `bioplausible/deployment.py` |
| CLI | `bioplausible/cli/` |
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
│   └── energy.py            # Energy profiling
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
│   │   ├── models.py        # LoopedMLP, StandardEqProp, DeepEP, ConvEqProp, ...
│   │   ├── propagators.py   # EqProp, HolomorphicEqProp, FiniteNudge, LazyEqProp
│   │   └── configs.py
│   ├── fa/
│   │   ├── __init__.py
│   │   ├── models.py        # FeedbackAlignmentModel, DirectFAModel, ...
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
│   │   ├── models.py        # FabricPCGraphPCN
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
│       └── inspector.py
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
├── deployment.py            # ONNX/TorchScript export, inference server
├── cli/                     # CLI (run.py, lab.py, rank.py)
├── datasets.py              # Data loading
├── domains/                 # Domain/task definitions
├── graph/                   # FabricPC graph API (implementation detail)
├── utils.py
├── visualization.py
├── visualization_tools.py
└── statistics.py
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
    LoopedMLP, StandardEqProp, DeepEP, ConvEqProp, EqPropDiffusion,
    TransformerEqProp, CausalTransformerEqProp,
    EqPropAttentionOnlyLM, FullEqPropLM, HybridEqPropLM,
    RecurrentEqPropLM, LoopedMLPForLM, MemoryEfficientLoopedMLP,
    NeuralCube, HomeostaticEqProp, TemporalResonanceEqProp,
    TernaryEqProp, SparseEquilibrium, MomentumEquilibrium,
)

# Propagators (imported from propagators.py)
from .propagators import (
    EqProp, HolomorphicEqProp, FiniteNudgeEqProp, LazyEqProp,
)

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

## 8. Import Migration Map (Complete)

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
| `from bioplausible.training.supervised import SupervisedTrainer` | `from bioplausible.core.trainer import CoreTrainer` (or `CoreTrainer` directly) |
| `from bioplausible.core import EqPropTrainer` | **DELETE** — was alias for SupervisedTrainer |
| `from bioplausible.hybrid_optimizer import HybridEqPropOptimizer` | **DELETE** — prototype |
| `from bioplausible.pipeline.config import TrainingConfig` | `from bioplausible.core.trainer import TrainerConfig` |
| `from bioplausible.pipeline.session import TrainingSession, SessionState` | `from bioplausible.core.trainer import CoreTrainer` (session pattern removed) |
| `from bioplausible.pipeline.events import ...` | **DELETE** — event system removed |
| `from bioplausible.pipeline.results import ResultsManager` | **DELETE** — results in CoreTrainer history |
| `from bioplausible.models.equitile import EquiTile, ConvEquiTile, ...` | `from bioplausible.equitile import EquiTile, ConvEquiTile, ...` |
| `from bioplausible.validation.tracks import core_tracks, scaling_tracks, ...` | `from bioplausible.validation.tracks import TrackRegistry` |

---

## 9. Execution Phases

### Phase 1: Documentation Archive & README (1 day)
1. Create `docs/archive/20260722/`
2. Move all files per §1.1
3. Write complete `README.md` per §1.2
4. Verify all links resolve

### Phase 2: Delete Dead Code (0.5 day)
1. Delete `bioplausible/compat.py`
2. Delete `bioplausible/models/registry.py`, `models/factory.py`
3. Delete `bioplausible/optimizers/learning_rules.py`, `optimizers/base.py`, `optimizers/__init__.py`
4. Delete `bioplausible/pipeline/`
5. Delete `bioplausible/training/supervised.py`, `training/base.py`
6. Delete `bioplausible/core.py`, `bioplausible/hybrid_optimizer.py`
6. Delete `bioplausible/models/tile_eq.py`
7. Delete `bioplausible/zoo/models/registered_models.py`, `zoo/propagators/registered_propagators.py`, `zoo/optimizers/registered_optimizers.py`, `zoo/sparsity/registered_sparsity.py`
8. Delete `bioplausible/scientist/` (entire dir — will be recreated as `execution/`)
9. Delete 8 redundant validation track files
10. Delete `/mep/` root (keep `/mep/mep/` contents for move)

### Phase 3: Move & Restructure (2 days)
1. Move `/mep/mep/` → `bioplausible/zoo/mep/`
2. Move `bioplausible/models/equitile/` → `bioplausible/equitile/`
3. Create algorithm-family dirs: `zoo/eqprop/`, `zoo/fa/`, `zoo/hebbian/`, `zoo/forward_only/`, `zoo/target_prop/`, `zoo/spiking/`, `zoo/predictive_coding/`, `zoo/backprop/`
4. Move model implementations from `models/*.py` into appropriate family `models.py`
5. Move propagator implementations from `optimizers/learning_rules.py` into family `propagators.py`
6. Create `execution/` dir with renamed files from `scientist/`
7. Consolidate validation tracks to 9 files + `TrackRegistry`
8. Update `pyproject.toml` packages list

### Phase 4: Register Everything (1 day)
1. Each family `__init__.py` registers models + propagators with rich metadata
2. `equitile/__init__.py` registers all EquiTile variants
3. `zoo/mep/__init__.py` registers presets as propagators, strategies as optimizers
4. `validation/tracks/__init__.py` creates TrackRegistry, registers tracks
5. Update `bioplausible/__init__.py` with clean public API

### Phase 5: Update All Consumers (1.5 days)
1. `cli/run.py`, `cli/lab.py` → Zoo registry + CoreTrainer
2. `hyperopt/` → Zoo registry for model/optimizer/propagator lookup
3. `execution/` (was scientist) → Zoo registry + CoreTrainer
4. `lightning_/` → Zoo registry for optimizers
5. `autoscientist/bridge.py` → Zoo registry for component discovery
6. `core/trainer.py` → Zoo registry for model/optimizer lookup
7. `examples/` → Updated imports
8. `tests/` → Updated imports (fix all test files)
9. `bioplausible_ui/` → **ARCHIVE ENTIRE DIRECTORY** (per user request)

### Phase 6: Format, Lint, Test (0.5 day)
1. `isort . && black .`
2. `flake8`
3. `pytest tests/` — fix failures
4. Verify `README.md` completeness

---

## 10. UI Code: Archive Entirely

```
MOVE → docs/archive/20260722/bioplausible_ui/:
  /home/me/bioplausible/bioplausible_ui/ (entire directory)
```

**Rationale**: Not complete, not used, separate package. Archive as-is for potential future resurrection.

---

## 11. Risk Assessment

| Risk | Mitigation |
|------|------------|
| External user imports break | **Accepted** — v1.0.0, no compat layer |
| Tests fail en masse | Phase 6 fixes; most are import updates |
| AutoScientist campaigns break | Safe — campaigns use DB, not imports |
| Forgotten import in obscure file | `grep -r` patterns in Phase 5 checklist |
| `bioplausible_ui` needs updates | Archived — not a concern |

---

## 12. Success Criteria

1. **Single docs entry**: `README.md` complete, only file new users need
2. **Single registry**: `Registry` in `core/registry.py` = source of truth
3. **Algorithm-family org**: `zoo/eqprop/`, `zoo/fa/`, `zoo/mep/`, `equitile/` — clear ownership
4. **No redundancy**: Each algorithm implemented once, registered once, documented once
5. **Clear naming**: Propagator=credit assignment, Optimizer=parameter update, Engine=execution, AutoScientist=LLM reasoner
6. **All tests pass**: No regressions
7. **~60 docs archived, ~15 Python files deleted, 2 registries → 1, 1 top-level package eliminated, UI archived**

---

## 13. File Count Impact (Estimated)

| Category | Before | After | Delta |
|----------|--------|-------|-------|
| Root `.md` files | 12 | 5 | -7 |
| `/docs/` `.md` files | 43 | 0 (archived) | -43 |
| `/mep/` (root) | 1 package | 0 | -1 |
| `bioplausible/compat.py` | 10K lines | 0 | -10K |
| `bioplausible/models/registry.py` + `factory.py` | 2 files | 0 | -2 |
| `bioplausible/optimizers/learning_rules.py` + `base.py` + `__init__.py` | 3 files | 0 | -3 |
| `bioplausible/pipeline/` | 4 files | 0 | -4 |
| `bioplausible/training/supervised.py` + `base.py` | 36K+ lines | 0 | -36K |
| `bioplausible/core.py` + `hybrid_optimizer.py` | 2 files | 0 | -2 |
| `bioplausible/models/tile_eq.py` | 1 file | 0 | -1 |
| `bioplausible/zoo/*/registered_*.py` | 4 files | 0 | -4 |
| `bioplausible/scientist/` | 1 dir | 0 (→ `execution/`) | 0 |
| Validation track files | 20 | 9 | -11 |
| `bioplausible_ui/` | 1 dir | 0 (archived) | -1 |

**Net**: ~46K lines deleted, cleaner architecture, single source of truth.