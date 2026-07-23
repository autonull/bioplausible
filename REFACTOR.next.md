# REFACTOR.next.md — Second Safe Cleanup Pass

**Goal**: Archive remaining dead/stale files identified in REFACTOR.md re-validation that have ZERO active imports and are NOT in the "KEEP" list. No import migrations, no functional changes.

**Time estimate**: 30 minutes

**Prerequisite**: REFACTOR.start.md completed and verified (tests pass).

---

## Scope: What We DO

| Item | Action | Rationale |
|------|--------|-----------|
| `REFACTOR.prompt.md` (root) | → `docs/archive/20260722/` | Stale prompt file, not in whitelist |
| `generate_report.sh` (root) | → `docs/archive/20260722/` | Not in §1.1 keep list; utility script |
| `mep/INTEGRATION_GUIDE.md` | → `docs/archive/20260722/mep/` | Superseded by README |
| `mep/PHASE2_FINAL_SUMMARY.md` | → `docs/archive/20260722/mep/` | Superseded by README |
| `bioplausible/experiments/README.md` | → `docs/archive/20260722/` | Experiment docs, not reusable infra |
| `bioplausible/experiments/LM_SCALE_STUDY.md` | → `docs/archive/20260722/` | Experiment docs |
| `bioplausible/cli.py` (root) | → `docs/archive/20260722/` | **Audit first**: if zero imports → archive; else keep |
| `docs/tutorials/` (entire dir) | → `docs/archive/20260722/tutorials/` | Superseded by future README; no code imports |

**Total**: ~10 files/dirs archived, ~0 lines of active code removed.

---

## Scope: What We DO NOT

- ❌ No `DELETE` of any Python module (all have active imports)
- ❌ No import migrations
- ❌ No directory restructuring
- ❌ No test/example/script updates
- ❌ No `mep/` package removal (Phase 4)

---

## Pre-Cleanup Audit: `bioplausible/cli.py`

Run this to confirm zero imports before archiving:

```bash
grep -r "from bioplausible.cli import\|import bioplausible.cli" --include="*.py" .
# If NO results → safe to archive
```

If imports exist → keep at root, do not archive.

---

## Verification Steps (after cleanup)

```bash
# 1. Archive created
ls docs/archive/20260722/REFACTOR.prompt.md
ls docs/archive/20260722/generate_report.sh
ls docs/archive/20260722/mep/INTEGRATION_GUIDE.md
ls docs/archive/20260722/mep/PHASE2_FINAL_SUMMARY.md
ls docs/archive/20260722/bioplausible_cli.py  # or cli.py
ls docs/archive/20260722/tutorials/

# 2. Legacy imports still resolve
python -c "
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible.optimizers import create_optimizer
from bioplausible.training.supervised import SupervisedTrainer
from bioplausible.scientist import Scientist
print('All legacy imports OK')
"

# 3. Core functionality intact
pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

---

## Rollback

```bash
mv docs/archive/20260722/REFACTOR.prompt.md .
mv docs/archive/20260722/generate_report.sh .
mv docs/archive/20260722/mep/INTEGRATION_GUIDE.md mep/
mv docs/archive/20260722/mep/PHASE2_FINAL_SUMMARY.md mep/
mv docs/archive/20260722/bioplausible_experiments_README.md bioplausible/experiments/README.md
mv docs/archive/20260722/bioplausible_experiments_LM_SCALE_STUDY.md bioplausible/experiments/LM_SCALE_STUDY.md
mv docs/archive/20260722/bioplausible_cli.py bioplausible/cli.py
mv docs/archive/20260722/tutorials/ docs/
```

---

## Execution Commands

```bash
# 1. Audit cli.py first
grep -r "from bioplausible.cli import\|import bioplausible.cli" --include="*.py" .

# 2. If clean, execute moves
mkdir -p docs/archive/20260722/tutorials

mv REFACTOR.prompt.md docs/archive/20260722/
mv generate_report.sh docs/archive/20260722/
mv mep/INTEGRATION_GUIDE.md mep/PHASE2_FINAL_SUMMARY.md docs/archive/20260722/mep/
mv bioplausible/experiments/README.md bioplausible/experiments/LM_SCALE_STUDY.md docs/archive/20260722/
mv bioplausible/cli.py docs/archive/20260722/bioplausible_cli.py
mv docs/tutorials/ docs/archive/20260722/tutorials/

# 3. Verify
python -c "from bioplausible.models.registry import MODEL_REGISTRY; print('OK')"
pytest tests/ -x -q --tb=short 2>&1 | tail -5
```