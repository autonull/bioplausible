# REFACTOR.start.md — Minimal Cleanup (Safe First Pass)

**Goal**: Execute a small, verifiable cleanup that removes *only* dead/obsolete code and archives unused docs — no API changes, no import migrations, no structural reorganization. After this, re-validate the full REFACTOR.md plan against the cleaned codebase.

**Time estimate**: 2-3 hours

---

## Scope: What We DO

| Item | Action | Rationale |
|------|--------|-----------|
| `/docs/*.md` (44 files) | → `docs/archive/20260722/` | Superseded by future README |
| Root `*.md` except `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `LICENSE`, `CHANGELOG.md`, `REFACTOR.md`, `REFACTOR.start.md` | → `docs/archive/20260722/` | `FABRICPC.plan.md`, `VERIFICATION.md`, `README0.md` are stale |
| `/mep/docs/*` (5 files) | → `docs/archive/20260722/mep/` | Stale |
| `/mep/README*.md` (2 files) | → `docs/archive/20260722/mep/` | Stale |
| `bioplausible_ui/` (entire dir) | → `docs/archive/20260722/bioplausible_ui/` | Not used, separate package |
| `research/` (2 entries) | → `docs/archive/20260722/research/` | Old research scripts |
| `benchmarks/` (root, 2 files) | → `docs/archive/20260722/benchmarks_root/` | Old benchmark scripts |
| `bioplausible/analysis_tools.py` | → `docs/archive/20260722/` | Overlaps with `analysis/` |
| `bioplausible/launch_studio.py` | → `docs/archive/20260722/` | UI-related |
| `bioplausible/run_equitile_ui.py` | → `docs/archive/20260722/` | UI-related |
| `bioplausible/verify.py` | → `docs/archive/20260722/` | One-off verification |
| `gui.sh`, `run_ui.sh`, `clear_scientist.sh` | → `docs/archive/20260722/` | UI/maintenance scripts |
| `bioplausible/models/benchmark.py` | → `docs/archive/20260722/` | Standalone benchmark |
| `bioplausible/models/nebc_base.py` | → `docs/archive/20260722/` | Abstract base, only 2-3 usages |
| `bioplausible/models/tile_eq.py` | **DELETE** | 2705L, superseded by `equitile/` |
| `asi_evolve/` | **DELETE** (already done) | Clone of separate codebase |

**Total**: ~80 files archived, 1 file deleted, ~3K lines removed. Zero functional changes.

---

## Scope: What We DO NOT

- ❌ No changes to `models/`, `optimizers/`, `training/`, `pipeline/`, `scientist/`, `core.py`, `compat.py`, `hybrid_optimizer.py`, `runner.py`
- ❌ No import migrations
- ❌ No registry changes
- ❌ No directory restructuring
- ❌ No test/example/script updates
- ❌ No README rewrite (yet)

---

## Verification Steps (after cleanup)

```bash
# 1. Archive created, files moved
ls docs/archive/20260722/
ls docs/archive/20260722/mep/
ls docs/archive/20260722/bioplausible_ui/

# 2. Legacy imports still resolve (nothing broke)
python -c "
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible.optimizers import create_optimizer
from bioplausible.training.supervised import SupervisedTrainer
from bioplausible.scientist import Scientist
print('All legacy imports OK')
"

# 3. Core functionality intact
pytest tests/ -x -q --tb=short 2>&1 | head -30

# 4. Smoke test
python smoke_test_all.py 2>&1 | tail -10
```

---

## If Verification Passes

Proceed to **re-validate full REFACTOR.md** against the cleaned codebase:
- Re-count files/lines
- Verify every row in Master Disposition Table still matches
- Confirm the remaining scope is achievable in phases

If verification fails → revert archive moves, debug, re-plan.

---

## Rollback

```bash
# If something breaks, restore from archive
mv docs/archive/20260722/* .
mv docs/archive/20260722/mep/* mep/ 2>/dev/null
mv docs/archive/20260722/bioplausible_ui/ .
# etc.
```