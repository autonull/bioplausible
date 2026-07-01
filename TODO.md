**Bioplausible Framework Preparation Specification**  
**Version:** 1.0 (June 2026)  
**Goal:** Transform the current rich but fragmented codebase into a unified, production-grade, autonomous research platform capable of systematic discovery across **all domains** (Language Modeling, Vision, Reinforcement Learning, Graphs, Time Series, Scientific Simulation, Tabular, Continual/Multi-Task, etc.).

### 1. Core Design Principles
- **Unified Zoo + Declarative Core**: Everything (models, propagators, optimizers, update rules, metrics, data loaders) lives in a single, extensible registry.
- **Composability First**: Any valid combination of components should train end-to-end via config.
- **Autonomy with Oversight**: Scientist = execution engine. AutoScientist = intelligent meta-reasoner (they are distinct and should remain so).
- **Reproducibility & Science**: Every run produces traceable artifacts; knowledgebase grows with every experiment.
- **Scalability**: Works on laptop → multi-node → future neuromorphic.
- **Extensibility**: Domain experts can add new tasks/rules with minimal friction.

### 2. Scientist vs AutoScientist — Clarification & Separation
- **Scientist** (current `bioplausible/scientist/`): The **autonomous execution & orchestration engine**. Handles task queuing, resource management, checkpointing, trial running, monitoring, failure recovery, and basic strategy selection (smoke/shallow/deep). It is rule-based + deterministic where possible.
- **AutoScientist**: The **LLM-augmented meta-cognitive layer**. Ingests logs + knowledgebase, proposes novel hypotheses, performs high-level reasoning, symbolic analysis, and generates intelligent experiment batches. It is the "scientist in the loop" that evolves understanding.

**Recommendation**: Keep them separate but tightly integrated. Scientist executes; AutoScientist decides what to execute next. This avoids bloating the reliable executor with brittle LLM calls.

Do **not** merge them completely — the distinction improves robustness.

### 3. Drastic but High-Value Refactorings
1. **Single Source of Truth Package** (`bioplausible/`):
   - Flatten and consolidate `mep/`, research subdirs, and scattered modules.
   - All public API exported from `bioplausible/__init__.py`.

2. **Registry System** (Central New Component):
   - Decorator-based + YAML-backed registry.
   - Every component registers with metadata: `domain_support`, `locality_level`, `compute_profile`, `bio_plausibility_score`, `credit_assignment_type`, etc.
   - Enables AutoScientist to query and compose intelligently.

3. **CoreTrainer** (New unifying class):
   - Replaces multiple runners. Accepts a config dict/YAML/OmegaConf specifying `model`, `propagator`, `optimizer`, `data`, `trainer_args`.
   - Uses Lightning under the hood for distributed but provides a clean local-first API.

4. **KnowledgeBase** (Upgrade):
   - Move to SQLite + vector store (or DuckDB) for hybrid structured + embedding search.
   - Integrate surrogate models, symbolic regression, causal discovery.

5. **Domain Abstraction Layer**:
   - `domains/` with standard interfaces (`Task`, `Dataset`, `Evaluator`, `MetricSuite`).
   - Makes adding new domains trivial.

### 4. Target Package Structure (Post-Refactor)
```
bioplausible/
├── __init__.py
├── core/                  # CoreTrainer, Registry, Config
├── zoo/                   # All components with @register decorators
│   ├── models/
│   ├── propagators/       # EqProp, FA, Hebbian, FF, etc.
│   ├── optimizers/
│   ├── sparsity/
│   └── ...
├── domains/               # vision, lm, rl, graph, timeseries, tabular, ...
├── training/              # Unified harness
├── data/                  # Loaders, curricula
├── evaluation/            # Standardized benchmarks
├── scientist/             # Execution engine (refactored)
├── autoscientist/         # New: LLM reasoning + proposal layer
├── knowledge/             # KnowledgeBase, surrogates, analysis
├── lightning_/            # Integration layer (keep but slim)
├── utils/
├── cli/
└── config/                # Schemas, defaults
```

### 5. Detailed Component Specifications

**Registry**:
- `@register_component(category="model", name="FastLMEquiTile", domains=["lm"], ...)` 
- Metadata enables constraint satisfaction (e.g., "only local rules for bio-plausible track").

**CoreTrainer**:
- `trainer = CoreTrainer(config)`
- `trainer.fit()` or `trainer.search()` (Optuna integration)
- Automatic Lightning wrapper, callbacks (energy, convergence, pruning), export.

**Scientist** (Execution Engine):
- Task queue with priorities
- Resource manager (GPU, memory, node awareness)
- Checkpointing + resumption
- Failure detection & retry with exponential backoff
- Integration with P2P/distributed

**AutoScientist**:
- Ingests experiment history
- Queries knowledgebase
- Proposes batches (with justifications)
- Uses LLM (optional local/open model) for high-level reasoning
- Human approval gate for expensive runs

**KnowledgeBase**:
- Stores: raw metrics, configs, artifacts pointers, surrogate predictions, extracted principles.
- Query API: natural language + structured.

**Domain Support**:
- Standardized `DomainTask` interface with `train_dataloader`, `val_dataloader`, `evaluate(model)` returning a rich metrics dict.
- Pre-built suites for major benchmarks per domain.

### 6. Preparation Roadmap (Phased)

**Phase 0: Foundation (2–4 weeks)**
- Implement Registry + CoreTrainer
- Migrate key models/propagators into zoo
- Unify experiment runners
- Basic KnowledgeBase v1 (SQLite + JSON export)

**Phase 1: Unification & Polish (3–5 weeks)**
- Refactor scientist module for robustness
- Standardize all domains
- Comprehensive test suite + CI
- CLI + GUI updates
- Documentation & examples refresh

**Phase 2: Intelligence Layer (4–6 weeks)**
- Build AutoScientist on top of Scientist + KnowledgeBase
- Add surrogate models & symbolic regression
- LLM integration (local-first)

**Phase 3: Validation & Release (COMPLETE)**
- Cross-domain benchmark suite ✅ (`bioplausible.evaluation.cross_domain`)
- Public leaderboard ✅ (`bioplausible.leaderboard.generator` + auto-integration)
- Contribution templates ✅ (`.github/` templates + `docs/CONTRIBUTING_DOMAIN.md`)
- Paper / tech report ✅ (`docs/PHASE3_REPORT.md`)

### 7. Non-Functional Requirements
- **Type Hints + Docstrings** everywhere (Google style)
- **Formatting**: Black + isort + flake8
- **Testing**: High coverage, including integration tests for Zoo combinations
- **Config**: OmegaConf with strict schema validation (Pydantic)
- **Logging & Observability**: Structured logs + dashboard
- **Licensing & Openness**: Keep MIT; clear contribution guidelines

### 8. Success Criteria
- Any registered combination trains with `python -m bioplausible train --config my.yaml`
- AutoScientist can run multi-day autonomous campaigns with minimal intervention
- Adding a new domain/rule takes < 1 day for an experienced contributor
- KnowledgeBase can answer "what works for X domain and why?"
- Reproducible results with one command

This specification preserves **all existing functionality** while dramatically increasing coherence, automation potential, and research velocity across every domain.

