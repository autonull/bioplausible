# REFACTOR2.md — Addendum: Completing the Reorganization Plan

This addendum layers on top of `REFACTOR.md`, filling gaps and correcting process issues. It does **not** repeat what `REFACTOR.md` already covers well (algorithm-family layout, registry pattern, EquiTile promotion, MEP relocation, validation consolidation, import migration map). Read both documents together.

---

## §A. Phase Ordering Correction

**Problem**: `REFACTOR.md` Phase 2 says "Delete Dead Code" **before** Phase 3 "Move & Restructure". But the files slated for deletion in Phase 2 (e.g. `models/*.py`, `optimizers/learning_rules.py`) contain the actual implementations needed for the new `zoo/*/models.py` and `zoo/*/propagators.py` files.

**Fix**: Merge and reorder the phases:

```
Phase 1: Documentation Archive & README          (unchanged from REFACTOR.md)
Phase 2: Create New Structure & Move Contents    (was Phase 3)
Phase 3: Register Everything                     (was Phase 4)
Phase 4: Delete Deprecated Files & Directories   (was Phase 2 — must come AFTER moves)
Phase 5: Update All Consumers                    (was Phase 5)
Phase 6: Format, Lint, Test                      (was Phase 6)
```

**Phase 4 (Delete) must not run until all content has been extracted to new locations.**

---

## §B. Complete Package Map (Expanding REFACTOR.md §2.1)

The following existing packages are **not mentioned** in `REFACTOR.md`'s target layout. They are all live, imported code used by 100+ files. Each must have an explicit destination.

### B.1 Packages That Stay as-Is (already well-structured)

| Package | Lines | Purpose | Keep as |
|---------|-------|---------|---------|
| `bioplausible/config/` | 3 files | OmegaConf/Pydantic config schemas | `bioplausible/config/` |
| `bioplausible/data/` | 4 files | Data loading (vision, LM, curricula) | `bioplausible/data/` |
| `bioplausible/domains/` | 9 files | DomainTask definitions (Vision, LM, RL, Graph, etc.) | `bioplausible/domains/` |
| `bioplausible/acceleration/` | 5 files | Hardware acceleration (triton, compile, backends) | `bioplausible/acceleration/` |
| `bioplausible/evaluation/` | 4 files | MetricSuite, EvaluatorBase, BenchmarkRegistry | `bioplausible/evaluation/` |
| `bioplausible/knowledge/` | 4 files | KnowledgeBase for experiment metadata | `bioplausible/knowledge/` |
| `bioplausible/leaderboard/` | 2 files | LeaderboardGenerator | `bioplausible/leaderboard/` |
| `bioplausible/graph/` | 6 files | FabricPC graph API (implementation detail) | `bioplausible/graph/` |
| `bioplausible/lightning_/` | 7 files | PyTorch Lightning integration | `bioplausible/lightning_/` |
| `bioplausible/p2p/` | 8 files | P2P coordinator system | `bioplausible/p2p/` |
| `bioplausible/cli/` | 5 files | CLI entry points | `bioplausible/cli/` |
| `bioplausible/hyperopt/` | 15 files | Optuna integration | `bioplausible/hyperopt/` |
| `bioplausible/autoscientist/` | 4 files | LLM reasoner | `bioplausible/autoscientist/` |
| `bioplausible/validation/` | tracks/ + core.py | Validation framework | `bioplausible/validation/` |

### B.2 Packages That Need Homes

#### `bioplausible/experiments/` (23+ files — 2 reusable + 21+ one-off scripts)

This is the **Experiment Runner** referenced in README. It has two layers:

**Keep and consolidate into `bioplausible/experiments/`**:
| File | Lines | Purpose |
|------|-------|---------|
| `utils.py` | 601 | `ExperimentRunner`, `HyperparameterSearch`, `quick_comparison`, `benchmark_model` |
| `presets.py` | 513 | `ResearchPreset`, category-based preset discovery |

These are the reusable infrastructure. They import from `models.registry`, `models.factory`, `optimizers` (old paths) — will need Phase 5 import updates.

**Archive to `docs/archive/20260722/experiments/`**:
All remaining ~21 scripts in `experiments/` that are one-off research runs:
```
adaptive_compute.py
benchmark_memory.py
benchmark_mnist.py
cifar_breakthrough.py
comprehensive_eval.py
deep_hebbian_mnist.py
deep_signal_probe.py
diffusion_mnist.py
energy_confidence.py
few_shot_study.py
final_characterization.py
flop_analysis.py
language_modeling_comparison.py
language_modeling.py
lm_scale_study.py
memory_scaling_demo.py
rl_comparison.py
run_all.py
shallow_search.py
sn_benchmark_datasets.py
sn_benchmark_model_size.py
sn_stress_test.py
track_a1_stability.py
```
These may reference old paths (models.registry, etc.) and are not maintained. Archive preserves them for reproducibility without cluttering the live codebase.

#### `bioplausible/analysis/` (6 files — analysis infrastructure)

| File | Lines | README Name | New Home |
|------|-------|-------------|----------|
| `results.py` | 365 | ResultAnalyzer | `bioplausible/analysis/results.py` |
| `scaling.py` | 119 | ScalingAnalyzer | `bioplausible/analysis/scaling.py` |
| `ablation.py` | 144 | AblationAnalyzer | `bioplausible/analysis/ablation.py` |
| `dynamics.py` | 251 | (TrainingVisualizer) | `bioplausible/analysis/dynamics.py` |
| `failure_manifesto.py` | 100 | FailureManifesto | `bioplausible/analysis/failure_manifesto.py` |
| `reporting.py` | 188 | (report generation) | `bioplausible/analysis/reporting.py` |

**Keep as `bioplausible/analysis/`** — these are well-structured already. Update imports from old paths (models.registry, scientist.*) after Phase 3/4.

---

## §C. Complete Top-Level File Map (Expanding REFACTOR.md §2.1)

The following loose `.py` files in `bioplausible/` are **not addressed** in REFACTOR.md's layout:

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `energy.py` | 93 | EnergyProfile dataclass + FLOPS counting | **Keep** — referenced by `core/energy.py` in REFACTOR target. Merge into `core/energy.py` |
| `export.py` | 123 | ONNX/TorchScript export + FastAPI inference server | **Merge** into `deployment.py` (same topic) |
| `generation.py` | 153 | Autoregressive text generation for LM models | **Keep** at `bioplausible/generation.py` |
| `kernel.py` | 997 | High-performance CPU/GPU kernel ops | **Move** to `bioplausible/acceleration/kernels.py` (replaces `acceleration/kernel.py` if duplicate) |
| `runner.py` | 136 | `run_from_config` — universal entry point | **Merge** into `core/trainer.py` as `CoreTrainer.run_from_config()` (already aliased) |
| `tracking.py` | 140 | WandB experiment tracking wrapper | **Keep** at `bioplausible/tracking.py` |
| `sklearn_interface.py` | 287 | Scikit-learn compatible wrapper for EqProp models | **Keep** at `bioplausible/sklearn_interface.py` |
| `analysis_tools.py` | 470 | Statistical comparison, effect sizes, reporting | **Deprecate** — overlaps with `analysis/` package. Archive to `docs/archive/20260722/` |
| `cli.py` | 135 | Additional CLI entry points | **Deprecate** — overlaps with `cli/` package. Audit, then archive or merge |
| `config_legacy.py` | 190 | Legacy GLOBAL_CONFIG, TRAINING_DEFAULTS | **Deprecate** — keep for now, mark as legacy in docstring, do NOT import from new code |
| `config_loader.py` | 52 | YAML config loader | **Merge** into `config/` package |
| `config_schema.py` | 62 | RunConfig pydantic schema | **Merge** into `config/schema.py` |
| `compat.py` | 391 | 10K-line backward compat | **Delete** (per REFACTOR.md) |
| `core.py` | 9 | Alias for EqPropTrainer | **Delete** (per REFACTOR.md) |
| `hybrid_optimizer.py` | 327 | HybridEqPropOptimizer prototype | **Delete** (per REFACTOR.md) |

---

## §D. Complete Model File → Family Mapping (Expanding REFACTOR.md §3)

REFACTOR.md says "move model implementations from `models/*.py` into appropriate family `models.py`" but does not list every file. Here is the complete map:

### D.1 Model Files

| File | Lines | Family Target | Notes |
|------|-------|---------------|-------|
| `looped_mlp.py` | ? | `zoo/eqprop/models.py` | LoopedMLP, BackpropMLP |
| `standard_eqprop.py` | ? | `zoo/eqprop/models.py` | StandardEqProp |
| `conv_eqprop.py` | ? | `zoo/eqprop/models.py` | ConvEqProp |
| `deep_ep.py` | ? | `zoo/eqprop/models.py` | DeepEP |
| `memory_efficient.py` | ? | `zoo/eqprop/models.py` | MemoryEfficientLoopedMLP |
| `transformer_eqprop.py` | ? | `zoo/eqprop/models.py` | TransformerEqProp |
| `causal_transformer_eqprop.py` | ? | `zoo/eqprop/models.py` | CausalTransformerEqProp |
| `eqprop_diffusion.py` | ? | `zoo/eqprop/models.py` | EqPropDiffusion |
| `holomorphic_ep.py` | ? | `zoo/eqprop/models.py` | HolomorphicEP |
| `finite_nudge_ep.py` | ? | `zoo/eqprop/models.py` | FiniteNudgeEP |
| `lazy_eqprop.py` | ? | `zoo/eqprop/models.py` | LazyEqProp |
| `neural_cube.py` | ? | `zoo/eqprop/models.py` | NeuralCube |
| `temporal_resonance.py` | ? | `zoo/eqprop/models.py` | TemporalResonanceEqProp |
| `ternary.py` | ? | `zoo/eqprop/models.py` | TernaryEqProp |
| `sparse_eq.py` | ? | `zoo/eqprop/models.py` | SparseEquilibrium |
| `mom_eq.py` | ? | `zoo/eqprop/models.py` | MomentumEquilibrium |
| `homeostatic.py` | ? | `zoo/eqprop/models.py` | HomeostaticEqProp |
| `modern_conv_eqprop.py` | ? | `zoo/eqprop/models.py` | ModernConvEqProp |
| `eqprop_lm_variants.py` | 700 | `zoo/eqprop/models.py` | FullEqPropLM, EqPropAttentionOnlyLM, RecurrentEqPropLM, HybridEqPropLM, LoopedMLPForLM |
| `graph_eqprop.py` | 148 | `zoo/eqprop/models.py` | GraphEqProp — **not in README, add it** |
| `feedback_alignment.py` | ? | `zoo/fa/models.py` | AdaptiveFeedbackAlignment |
| `dfa_eqprop.py` | ? | `zoo/fa/models.py` | DirectFeedbackAlignmentEqProp |
| `simple_fa.py` | 134 | `zoo/fa/models.py` | StandardFA |
| `eg_fa.py` | ? | `zoo/fa/models.py` | EnergyGuidedFA |
| `em_fa.py` | ? | `zoo/fa/models.py` | EnergyMinimizingFA |
| `leq_fa.py` | ? | `zoo/fa/models.py` | LayerwiseEquilibriumFA |
| `eq_align.py` | ? | `zoo/fa/models.py` | EquilibriumAlignment |
| `hebbian_chain.py` | ? | `zoo/hebbian/models.py` | DeepHebbianChain |
| `three_factor.py` | ? | `zoo/hebbian/models.py` | ThreeFactorHebbian |
| `chl.py` | ? | `zoo/hebbian/models.py` (or propagators) | ContrastiveHebbianLearning — straddles model/propagator boundary |
| `forward_forward.py` | ? | `zoo/forward_only/models.py` | Forward-Forward |
| `pepita.py` | ? | `zoo/forward_only/models.py` | PEPITA |
| `spiking_stdp.py` | ? | `zoo/spiking/models.py` | SpikingSTDP |
| `target_prop.py` | ? | `zoo/target_prop/models.py` | DifferenceTargetPropagation |
| `fabricpc_graph_pcn.py` | ? | `zoo/predictive_coding/models.py` | FabricPCGraphPCN |
| `backprop_transformer_lm.py` | ? | `zoo/backprop/models.py` | BackpropTransformerLM |
| `pc_hybrid.py` | 118 | `zoo/predictive_coding/models.py` | PredictiveCodingHybrid — **not in README, add it** |
| `custom_stack.py` | 204 | `zoo/backprop/models.py` or `utils.py` | Generic layer builder — utility, not an algorithm per se |

### D.2 Base Classes and Support Files

| File | Lines | Action |
|------|-------|--------|
| `eqprop_base.py` | ? | **Move** to `zoo/eqprop/` as `base.py` — EqProp model base class |
| `eqprop_wrappers.py` | 294 | **Move** to `zoo/eqprop/` as `wrappers.py` — RecurrentWrapper, etc. |
| `nebc_base.py` | 195 | **Archive** or **move** to `zoo/nebc_base.py` — abstract base, only 2-3 usages |
| `base.py` | ? | **Keep** at `zoo/base.py` — BioModel base used by StandardFA, PredictiveCodingHybrid, etc. |
| `tile_eq.py` | 2705 | **Delete** (per REFACTOR.md) — superseded by equitile/ |
| `benchmark.py` | 112 | **Archive** — standalone benchmark script, not a model |
| `utils.py` | ? | **Move** to `zoo/utils.py` or merge into `bioplausible/utils.py` |
| `triton_kernel.py` | ? | **Move** to `acceleration/triton_kernels.py` |

### D.3 Propagator Files (from `optimizers/`)

| File | Lines | Family Target |
|------|-------|---------------|
| `optimizers/learning_rules.py` | 814 | Split into `zoo/*/propagators.py` per family |
| `optimizers/base.py` | 35 | Delete (replaced by new structure) |
| `optimizers/__init__.py` | 138 | Delete |

---

## §E. README Completeness: Undocumented Implementations

The following models exist in the codebase but are **not listed** in `README.md`. After reorganization, the README must either add them or explicitly archive them:

### E.1 Models to ADD to README

| Model | File | Family | Description |
|-------|------|--------|-------------|
| `GraphEqProp` | `models/graph_eqprop.py` | EqProp | Graph-structured EqProp (a different model from GraphEquiTile) |
| `PredictiveCodingHybrid` | `models/pc_hybrid.py` | Predictive Coding | Combines PC top-down predictions with FA error signals |
| `StandardFA` | `models/simple_fa.py` | Feedback Alignment | Basic random-fixed-weight feedback alignment |
| `RecurrentWrapper` | `models/eqprop_wrappers.py` | EqProp | Generic wrapper turning any RNN cell into EqProp-compatible model |

### E.2 Support/Utility Files to Note in README

| File | README Section |
|------|----------------|
| `custom_stack.py` | Mention as utility for building custom architectures |
| `eqprop_base.py` / `wrappers.py` | Mention as base infrastructure under EqProp family |
| `generation.py` | Mention under "Deployment & Inference" → "Text Generation" |
| `sklearn_interface.py` | Mention under "Deployment & Inference" → "Scikit-learn Interface" |
| `tracking.py` | Mention under "Experiment Tracking" |

---

## §F. MEP Package: Complete Migration

REFACTOR.md moves `mep/mep/` → `zoo/mep/`. But the following MEP sub-packages are not addressed:

### F.1 `mep/mep/benchmarks/` (10 files)

| File | Action |
|------|--------|
| `baselines.py` | **Move** to `zoo/mep/benchmarks/baselines.py` |
| `compare.py` | **Move** to `zoo/mep/benchmarks/compare.py` |
| `metrics.py` | **Move** to `zoo/mep/benchmarks/metrics.py` |
| `runner.py` | **Move** to `zoo/mep/benchmarks/runner.py` |
| `config/` | **Move** to `zoo/mep/benchmarks/config/` |
| `niche_benchmarks.py` | **Move** to `zoo/mep/benchmarks/niche_benchmarks.py` |
| `continual_learning.py` | **Move** to `zoo/mep/benchmarks/continual_learning.py` |
| `ewc_baseline.py` | **Move** to `zoo/mep/benchmarks/ewc_baseline.py` |
| `tuned_compare.py` | **Move** to `zoo/mep/benchmarks/tuned_compare.py` |
| `visualization.py` | **Move** to `zoo/mep/benchmarks/visualization.py` |

### F.2 `mep/mep/cuda/` (2 files)

| File | Action |
|------|--------|
| `__init__.py` | **Move** to `zoo/mep/cuda/__init__.py` |
| `kernels.py` | **Move** to `zoo/mep/cuda/kernels.py` |

### F.3 `mep/mep/optimizers_legacy.py`

**Archive** — old prototype, superseded by `optimizers/` sub-package.

---

## §G. Additional Helpful Work

### G.1 Consolidate Duplicate Analysis Modules

`bioplausible/analysis_tools.py` (470 lines) overlaps substantially with `bioplausible/analysis/`. Audit before archival:
- `StatisticalComparison`, `effect_size`, `confidence_interval` — may have unique functionality not in `analysis/results.py`
- Extract any unique code into `analysis/results.py`, then archive `analysis_tools.py`

### G.2 Consolidate CLI

`bioplausible/cli.py` (135 lines) appears to be an older CLI entry point, while `bioplausible/cli/` (5 files) is the current one. Audit:
- If `cli.py` is dead, archive it
- If it contains unique commands, merge them into `cli/run.py`, `cli/lab.py`, or `cli/rank.py`

### G.3 Consolidate Config

`config_legacy.py`, `config_loader.py`, `config_schema.py` are legacy top-level files while `bioplausible/config/` (3 files) is the current config system.
- Move `config_loader.py` functionality into `config/__init__.py`
- Move `config_schema.py` (RunConfig) into `config/schema.py`
- Keep `config_legacy.py` but do NOT import from it in new code; mark it `# DEPRECATED` at the top

### G.4 Root-Level Shell Scripts

9 shell scripts at root: `clear_scientist.sh`, `generate_report.sh`, `gui.sh`, `lab.sh`, `run_benchmark.sh`, `run_benchmarks.sh`, `run_leaderboard.sh`, `run_scientist.sh`, `run_ui.sh`

- Archive `gui.sh`, `run_ui.sh` (UI-related — goes with `bioplausible_ui/` archival)
- Archive `clear_scientist.sh` (maintenance script)
- Keep the rest as-is (entry points) but update any internal paths after reorganization

### G.5 Root-Level Python Scripts

`launch_leaderboard.py`, `launch_studio.py`, `run_equitile_ui.py`, `smoke_test_all.py`, `test_formatting.py`

- Archive `launch_studio.py`, `run_equitile_ui.py` (UI-related)
- Keep `launch_leaderboard.py` (moves to `cli/` or scripts/)
- Keep `smoke_test_all.py`, `test_formatting.py` (testing)

### G.6 Root-Level Data/Runtime Directories (No Action Required)

These are runtime/generated artifacts, not source code. Do NOT move or delete:
- `autoscientist_campaigns/` — AutoScientist experiment output
- `benchmarks/` — benchmark results
- `benchmark_results/` — result data
- `checkpoints/` — model checkpoints
- `configs/` — user YAML configs
- `data/` — downloaded datasets
- `experiments/` — legacy experiment output
- `lightning_logs/` — PyTorch Lightning logs
- `logs/` — application logs
- `research/` — research notes/output
- `results/` — result data
- `screenshots/` — UI screenshots
- `bioplausible.db`, `dummy.db`, `bioplausible_kb.db` — sqlite databases
- `knowledgebase.json`, `test_kb.json` — knowledge base data
- Various `.json` result files in `/mep/`

### G.7 `models/__init__.py` (502 lines) Treatment

This file currently re-exports ~40 models with try/except guards. It must be **deleted** as part of `models/` directory removal, but its content should be audited first:
- Ensure every model it re-exports has a home in the new family structure
- The try/except patterns suggest some models are optional (missing deps). Preserve that pattern in the family `__init__.py` files

### G.8 Refactor `experiments/presets.py` to Use Zoo Registry

`presets.py` has hardcoded model/optimizer name strings:
```python
ResearchPreset(name="vision_eqprop", model_name="conv_eqprop", optimizer_name="smep", ...)
```
After reorganization, these should resolve through `Registry.get("model", name)` / `Registry.get("propagator", name)` so that new algorithms are automatically discoverable by the preset system.

### G.9 Deprecate `runner.py` in Favor of `CoreTrainer`

`runner.py`'s `run_from_config()` is a thin wrapper around old `models.create_model()` + `optimizers.create_optimizer()`. The new `core/trainer.py`'s `CoreTrainer.run_from_config()` already exists as the replacement. After Phase 5 consumer updates, `runner.py` can be deleted.

### G.10 Regularize Naming in `lightning_/`

The trailing underscore in `lightning_/` is unusual (to avoid clashing with the `lightning` package itself). Consider renaming to `lightning_integration/` or `lightning_pl/` for clarity, but this is optional — the underscore convention is standard Python for shadowed stdlib names.

---

## §H. Summary of File Count Impact (Corrected)

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

## §I. Additional Success Criteria

In addition to REFACTOR.md §12:

1. **No orphan imports**: `grep -r "from bioplausible.models"` and `from bioplausible.optimizers` return 0 hits after Phase 5
2. **All 3 undocumented models** (GraphEqProp, PredictiveCodingHybrid, StandardFA) are either in README or archived
3. **`experiments/presets.py`** resolves models through Zoo Registry, not hardcoded strings
4. **All packages in §B.1** have explicit `__init__.py` entries for their exports
5. **`plot_results.py`, `generate_report.sh` etc.** are verified to work after path migrations
6. **`mep/mep/benchmarks/`** runs correctly from new `zoo/mep/benchmarks/` location
