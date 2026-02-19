# EP Development Workflow

## Golden Rule

**ALWAYS run smoke tests AFTER code changes and BEFORE complex benchmarks.**

```
Code Change → Smoke Tests (<20s) → Full Regression (<3min) → Complex Benchmarks
              ↑                       ↑                        ↑
              |                       |                        |
         MUST PASS              Optional, more           Only if regression
                                thorough                  tests pass
```

---

## Smoke Test Suite (MANDATORY)

### Location
```
tests/regression/test_ep_smoke.py
```

### What It Tests

| Test | Purpose | Pass Criteria | Time |
|------|---------|---------------|------|
| MNIST Learning | EP actually learns | >15% accuracy (1 epoch, 10k samples) | ~12s |
| Deep Stability | No NaN/Inf at depth | 500 layers, no NaN/Inf | ~1s |
| Backward Compat | Presets still work | smep, smep_fast, muon_backprop | ~1s |

**Total runtime:** < 20 seconds

### How to Run

```bash
# After EVERY code change
python tests/regression/test_ep_smoke.py

# Exit code 0 = all passed, safe to continue
# Exit code 1 = regression detected, fix bug first
```

---

## Full Regression Suite (Optional, More Thorough)

### Location
```
tests/regression/test_ep_baseline.py
```

### What It Tests

| Test | Purpose | Pass Criteria | Time |
|------|---------|---------------|------|
| MNIST Learning | EP learns well | >55% accuracy (3 epochs, full data) | ~150s |
| Deep Stability | No NaN/Inf at depth | 1000 layers, no NaN/Inf | ~5s |
| Backward Compat | Presets still work | smep, smep_fast, muon_backprop | ~5s |

**Total runtime:** < 3 minutes

### When to Run

- Before committing major changes
- Before running complex benchmarks
- When smoke test passes but something feels wrong

---

## Workflow

### 1. After Code Changes

```bash
# ALWAYS run this first
python tests/regression/test_ep_baseline.py

# If PASS: continue to complex benchmarks
# If FAIL: fix bug, re-run regression tests
```

### 2. Before Complex Benchmarks

```bash
# Verify regression tests still pass
python tests/regression/test_ep_baseline.py && \
    python examples/complex_benchmark.py
```

### 3. CI/CD (Future)

Add to GitHub Actions:
```yaml
- name: Regression Tests
  run: python tests/regression/test_ep_baseline.py
```

---

## Known Working Configurations

### MNIST Classification (Baseline)

```python
from mep import smep

opt = smep(
    model.parameters(),
    model=model,
    lr=0.01,
    mode='ep',
    settle_steps=30,    # Critical: enough settling
    settle_lr=0.15,     # Critical: stable settling
    beta=0.5,           # Nudging strength
    loss_type='mse',    # Critical: mse works, cross_entropy needs tuning
)
```

**Expected:** 60% after 3 epochs, 90%+ after 10 epochs

### Fast Prototyping

```python
from mep import smep_fast

opt = smep_fast(
    model.parameters(),
    model=model,
    lr=0.01,
    settle_steps=10,    # Faster, less accurate
    settle_lr=0.2,
)
```

**Expected:** 4-6x faster than default, slightly lower accuracy

### Backprop Comparison

```python
from mep import muon_backprop

opt = muon_backprop(
    model.parameters(),
    lr=0.02,
)
```

---

## Critical Parameters

### Must Match Known Working Values

| Parameter | Working Value | Effect if Wrong |
|-----------|---------------|-----------------|
| `loss_type` | `'mse'` | `cross_entropy` currently broken |
| `settle_steps` | 30 | Too few = no learning |
| `settle_lr` | 0.15 | Too high/low = no convergence |
| `beta` | 0.5 | Affects nudging strength |

### Safe to Experiment

| Parameter | Range | Effect |
|-----------|-------|--------|
| `lr` | 0.001-0.02 | Main learning rate |
| `momentum` | 0.9-0.95 | Momentum factor |
| `weight_decay` | 0.0001-0.001 | Regularization |

---

## Debugging Checklist

If regression tests fail:

1. **Check loss_type** - Must be `'mse'` (not `'cross_entropy'`)
2. **Check settle_steps** - Must be ≥30 for learning
3. **Check settle_lr** - Must be ~0.15 (not 0.2)
4. **Check model init** - Must use kaiming_normal for deep nets
5. **Check data preprocessing** - Must normalize MNIST

---

## Version Control

### Before Committing

```bash
# Run regression tests
python tests/regression/test_ep_baseline.py

# If pass, commit
git add .
git commit -m "Description"

# If fail, DO NOT COMMIT
# Fix bug, re-run tests
```

### Before Merging PR

```bash
# Ensure main branch regression tests pass
git checkout main
python tests/regression/test_ep_baseline.py

# Test feature branch
git checkout feature-branch
python tests/regression/test_ep_baseline.py
```

---

## Historical Bugs (Don't Repeat)

### Bug #1: cross_entropy Path Broken (2026-02-19)

**Symptom:** MNIST accuracy stuck at 10% (random)

**Root Cause:** Internal energy used KL divergence instead of MSE

**Fix:** Internal energy is ALWAYS MSE. loss_type only affects nudge term.

**Lesson:** Test with known working config before complex benchmarks.

### Bug #2: Model Parameter Missing (2026-02-19)

**Symptom:** Backprop mode crashes without model

**Root Cause:** EWC state required model but backprop didn't pass it

**Fix:** Make model optional, only required for EP mode + EWC

**Lesson:** Test all code paths, not just happy path.

---

## Summary

**Rule:** Regression tests FIRST, complex benchmarks SECOND.

**Time investment:** 2 minutes per code change.

**Time saved:** Hours debugging complex benchmarks when the bug was fundamental.

---

*Created: 2026-02-19*
*Status: Active*
