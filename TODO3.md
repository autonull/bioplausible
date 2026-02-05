# TODO3: Scientist++ Stability, Reporting, & Infrastructure

> **Status**: Ready for Implementation  
> **Priority**: CRITICAL - Addresses failures, broken tasks, and missing features  
> **Effort**: 4-6 weeks (6 phases)  
> **Replaces**: TODO.md + TODO2.md

---

## Executive Summary

This document consolidates **critical stability fixes**, **comprehensive reporting improvements**, and **scientific infrastructure** into a unified implementation plan.

**Based on**:
- ✅ TODO2.md (completed, but reporting partially reverted - we'll restore the good parts)
- ✅ Report analysis (128 trials: 30% failure rate, broken tasks, shallow exploration)
- ✅ TODO.md (infrastructure and release readiness goals)

**Three Pillars**:
1. **Stability** (Weeks 1-2): Fix 30%+ failure rates, algorithm-specific constraints
2. **Reporting** (Weeks 3-4): Training dynamics, modular reports, AI synthesis
3. **Infrastructure** (Weeks 5-6): Experiment tracking, visualization, documentation

---

## Phase 1: Critical Stability Fixes (Weeks 1-2) 🔥

**Goal**: Reduce failure rate from 30%+ to <10%

### 1.1 Universal Safety System

**Problem**: NaN losses, exploding gradients, OOM errors

**Files to Create/Modify**:
- `bioplausible/hyperopt/safety.py` (NEW)
- `bioplausible/hyperopt/runner.py`

**Implementation**:

```python
# bioplausible/hyperopt/safety.py

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SafetyConfig:
    """Safety configuration for training."""
    max_grad_norm: float = 10.0
    nan_check_frequency: int = 10
    lr_reduction_on_nan: float = 0.5
    max_nan_retries: int = 3
    enable_anomaly_detection: bool = False

class SafetyWrapper:
    """Wraps training to catch and handle numerical instabilities."""
    
    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.consecutive_failures = 0
        
    def safe_backward_and_step(
        self, 
        loss: torch.Tensor, 
        optimizer,
        model,
        clip_norm: Optional[float] = None
    ) -> Tuple[bool, dict]:
        """
        Perform backward + step with safety checks.
        
        Returns:
            (success, info): success=True if step completed, info=metrics or error
        """
        # 1. Check loss validity
        if not torch.isfinite(loss):
            return False, {"error": "loss_nan_or_inf", "loss_value": float(loss)}
        
        # 2. Backward pass
        try:
            loss.backward()
        except RuntimeError as e:
            return False, {"error": "backward_failed", "exception": str(e)}
        
        # 3. Check gradients
        total_norm = 0.0
        has_nan = False
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                if not torch.isfinite(param_norm):
                    has_nan = True
                    break
                total_norm += param_norm.item() ** 2
        
        if has_nan:
            optimizer.zero_grad()
            return False, {"error": "grad_nan", "grad_norm": float('nan')}
        
        total_norm = total_norm ** 0.5
        
        # 4. Clip gradients
        clip_value = clip_norm or self.config.max_grad_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # 5. Step optimizer
        try:
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            return False, {"error": "optimizer_step_failed", "exception": str(e)}
        
        # Success
        self.consecutive_failures = 0
        return True, {"grad_norm": total_norm, "loss": float(loss)}
    
    def should_abort(self) -> bool:
        """Check if training should be aborted."""
        return self.consecutive_failures >= self.config.max_nan_retries
    
    def handle_failure(self, optimizer):
        """Handle a training failure."""
        self.consecutive_failures += 1
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.config.lr_reduction_on_nan
```

### 1.2 Algorithm-Specific Hyperparameter Constraints

**Problem**: EqProp using same LR range as Backprop → failures

**Files to Create/Modify**:
- `bioplausible/scientist/algorithm_constraints.py` (NEW)
- Integration into `bioplausible/scientist/core.py`

**Implementation**:

```python
# bioplausible/scientist/algorithm_constraints.py

from typing import Tuple, Dict

ALGORITHM_FAMILY_CONSTRAINTS = {
    "baseline": {
        "lr": (1e-5, 1e-2, "log"),
        "grad_clip": (0.5, 10.0, "linear"),
        "weight_decay": (0.0, 1e-2, "log"),
        "dropout": (0.0, 0.5, "linear"),
    },
    "eqprop": {
        "lr": (1e-6, 5e-4, "log"),  # Much lower for stability!
        "beta": (0.01, 0.5, "linear"),
        "steps": (10, 40, "int"),
        "grad_clip": (1.0, 5.0, "linear"),  # Tighter clipping
        # NO optimizer, dropout, weight_decay (not applicable)
    },
    "hebbian": {
        "lr": (1e-5, 1e-3, "log"),
        "contrastive_steps": (5, 30, "int"),
        "grad_clip": (1.0, 10.0, "linear"),
    },
    "hybrid": {
        "lr": (1e-5, 5e-3, "log"),
        "grad_clip": (0.5, 10.0, "linear"),
        # May include both FA and equilibrium params
    },
}

def get_constrained_search_space(model_name: str) -> Dict:
    """
    Returns algorithm-specific hyperparameter constraints.
    """
    from bioplausible.models.registry import get_model_spec
    
    model_spec = get_model_spec(model_name)
    family = model_spec.family
    
    return ALGORITHM_FAMILY_CONSTRAINTS.get(family, ALGORITHM_FAMILY_CONSTRAINTS["baseline"])

# Integration example:
# In create_optuna_trial():
#   constraints = get_constrained_search_space(model_name)
#   config["lr"] = trial.suggest_float("lr", *constraints["lr"][:2], log=(constraints["lr"][2]=="log"))
```

### 1.3 Failure Tracking & Diagnostics

**Files to Create**:
- `bioplausible/scientist/failure_tracker.py` (NEW)

**Schema**:
```sql
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    task_name TEXT NOT NULL,
    tier TEXT,
    trial_id INTEGER,
    failure_type TEXT NOT NULL,  -- "nan_loss", "grad_explode", "oom", "timeout"
    failure_epoch INTEGER,
    config TEXT,
    last_metrics TEXT,
    stack_trace TEXT
);
```

**CLI Integration**:
```bash
bioplausible-scientist failures --recent 24h --group-by type
bioplausible-scientist failures --model "EqProp MLP" --analyze
```

---

## Phase 2: Task Quality & Curriculum (Week 2)

**Goal**: Replace broken tasks, add progressive difficulty

### 2.1 Task Replacement

| Current | Issue | Replacement |
|---------|-------|-------------|
| `cartpole` | 8% best (random) | `pendulum` or `acrobot` |
| `tiny_shakespeare` | 16% best | `char_ngram` (simpler) |

**Files to Create**:
- `bioplausible/tasks/lm/char_ngram.py`
- `bioplausible/tasks/rl/pendulum.py`

### 2.2 Task-Specific Promotion Thresholds

**Problem**: Models promoted at 17-25% (near-random)

```python
# bioplausible/scientist/promotion.py (NEW)

def get_promotion_threshold(tier: PatientLevel, task: str) -> float:
    """Task-specific promotion thresholds."""
    
    # Tier-specific requirements
    THRESHOLDS = {
        PatientLevel.SMOKE: {
            "mnist": 0.60,          # Was 0.15 (too low!)
            "fashion_mnist": 0.55,
            "char_bigram": 0.24,    # Was accepting 17%
            "char_trigram": 0.25,
            "pendulum": -800,       # Reward-based (lower is better)
        },
        PatientLevel.SHALLOW: {
            "mnist": 0.75,
            "char_bigram": 0.40,
            "pendulum": -300,
        },
        PatientLevel.STANDARD: {
            "mnist": 0.85,
            "char_bigram": 0.55,
        },
        PatientLevel.DEEP: {
            "mnist": 0.92,
        },
    }
    
    return THRESHOLDS[tier].get(task, 0.50 if tier == PatientLevel.SMOKE else 0.80)
```

### 2.3 Curriculum Learning

**Files to Create**:
- `bioplausible/scientist/curriculum.py`

```python
VISION_CURRICULUM = [
    ("mnist", PatientLevel.STANDARD, 0.85),
    ("fashion_mnist", PatientLevel.STANDARD, 0.80),
    ("cifar10", PatientLevel.STANDARD, 0.70),
]

LANGUAGE_CURRICULUM = [
    ("char_bigram", PatientLevel.STANDARD, 0.40),
    ("char_trigram", PatientLevel.STANDARD, 0.45),
    ("char_5gram", PatientLevel.STANDARD, 0.50),
    ("tiny_shakespeare", PatientLevel.STANDARD, 0.60),
]
```

---

## Phase 3: Training Dynamics & Checkpointing (Week 3)

**Goal**: Capture full learning trajectories instead of just final accuracy

### 3.1 Progressive Checkpoint System

**Problem**: Only recording final accuracy → missing convergence/overfitting signals

**Files to Create**:
- `bioplausible/scientist/training_dynamics.py`

**Implementation**:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingCheckpoint:
    """Metrics at a single checkpoint."""
    epoch: int
    train_acc: float
    val_acc: float
    test_acc: Optional[float]
    train_loss: float
    val_loss: float
    
    # Training dynamics
    grad_norm_mean: float
    grad_norm_std: float
    weight_norm: float
    learning_rate: float
    
    # Overfitting
    train_val_gap: float  # train_acc - val_acc
    
    # Task-specific
    perplexity: Optional[float] = None  # LM
    reward: Optional[float] = None      # RL
    
    # Efficiency
    wall_time_seconds: float = 0.0

@dataclass
class TrainingTrajectory:
    """Complete training history."""
    trial_id: int
    model_name: str
    task_name: str
    config: dict
    checkpoints: List[TrainingCheckpoint]
    
    # Derived metrics
    convergence_epoch: Optional[int] = None
    converged: bool = False
    overfitting_detected: bool = False
    unstable: bool = False
    
    def compute_sample_efficiency(self) -> float:
        """Area under learning curve (AUC)."""
        epochs = [c.epoch for c in self.checkpoints]
        accs = [c.val_acc for c in self.checkpoints]
        return np.trapz(accs, epochs) / epochs[-1] if epochs else 0.0

# Standard checkpoints: [1, 2, 5, 10, 20, 50, 100]
```

**Database Schema**:
```sql
CREATE TABLE IF NOT EXISTS training_trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id INTEGER NOT NULL,
    model_name TEXT,
    task_name TEXT,
    config JSON,
    convergence_epoch INTEGER,
    converged BOOLEAN,
    overfitting_detected BOOLEAN,
    FOREIGN KEY (trial_id) REFERENCES trials(id)
);

CREATE TABLE IF NOT EXISTS training_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id INTEGER NOT NULL,
    epoch INTEGER,
    train_acc REAL,
    val_acc REAL,
    train_loss REAL,
    val_loss REAL,
    grad_norm_mean REAL,
    weight_norm REAL,
    train_val_gap REAL,
    perplexity REAL,
    reward REAL,
    wall_time_seconds REAL,
    FOREIGN KEY (trajectory_id) REFERENCES training_trajectories(id)
);
```

---

## Phase 4: Modular Report Architecture (Week 4)

**Goal**: Replace monolithic index.md with composable, AI-analyzable reports

### 4.1 Multi-File Report Structure

**Current Problem**: Single 25KB index.md, hard for AI to parse

**Solution**:
```
reports/
  run_2026-02-05_10-30-00/
    manifest.json              # Report metadata
    
    # Core sections
    01_summary.md              # Executive summary
    02_discoveries.md          # Key findings (auto-generated)
    03_leaderboards.md         # Task rankings
    04_training_dynamics.md    # Learning curves, convergence
    05_hyperparameter_analysis.md  # Decision trees, importance
    06_statistical_tests.md    # Significance matrices
    
    # Per-task deep dives
    tasks/
      mnist_analysis.md
      char_bigram_analysis.md
    
    # Per-algorithm deep dives
    algorithms/
      eqprop_mlp_analysis.md
      finite_nudge_eqprop_analysis.md
    
    # Visualizations
    images/
      leaderboard_mnist.png
      curves_eqprop_family.png
      pareto_frontier.png
    
    # Compilation targets
    FULL_REPORT.md            # All sections combined
    AI_ANALYSIS_READY.md      # Optimized for LLM input
```

### 4.2 Report Composer

**Files to Create**:
- `bioplausible/scientist/report/composer_v2.py`

```python
class ReportComposer:
    """Compose modular reports."""
    
    def compose_full_report(self) -> str:
        """Concatenate all sections."""
        sections = [
            "01_summary.md",
            "02_discoveries.md",
            "03_leaderboards.md",
            "04_training_dynamics.md",
            "05_hyperparameter_analysis.md",
            "06_statistical_tests.md",
        ]
        return self._concatenate_files(sections)
    
    def compose_ai_analysis_input(self) -> str:
        """
        Generate AI-optimized report.
        Prepend metadata as structured JSON.
        """
        metadata_str = json.dumps(self.manifest, indent=2)
        preamble = f"""# AI Analysis Dataset\n\n## Metadata\n```json\n{metadata_str}\n```\n"""
        return preamble + self._concatenate_files(["02_discoveries.md", "05_hyperparameter_analysis.md"])
```

---

## Phase 5: AI-Powered Research Synthesis (Week 4-5)

**Goal**: Auto-generate insights, recommendations, and quick wins

### 5.1 Synthesis Engine

**Files to Restore/Enhance**:
- `bioplausible/scientist/synthesizer.py`

**Already implemented** (from TODO2), but was generating minimal output. Enhance:

```python
class ResearchSynthesizer:
    """Enhanced synthesis with failure analysis."""
    
    def synthesize_full_report(self) -> Dict:
        return {
            "cross_algorithm_insights": self.generate_cross_algorithm_insights(),
            "architectural_recommendations": self.generate_architecture_recommendations(),
            "research_gaps": self.identify_research_gaps(),
            "actionable_quick_wins": self.find_quick_wins(),
            "failure_analysis": self.analyze_failures(),  # NEW
            "curriculum_suggestions": self.suggest_curriculum(),  # NEW
        }
    
    def analyze_failures(self) -> Dict:
        """Correlate failures with hyperparameters."""
        # Which configs lead to NaN?
        # Which models fail most often on which tasks?
        # Suggest constraint tightening
        pass
```

### 5.2 Example Synthesis Output

```markdown
## Cross-Algorithm Insights

### Sample Efficiency
**MNIST**: hebbian (AUC: 0.87) > eqprop (0.79) > baseline (0.71)  
**Interpretation**: Hebbian learns efficiently from few samples.

## Architectural Recommendations

### 🔬 Recommendation #1: Hybrid EqProp-Backprop Transformer
**Priority**: ⭐⭐⭐⭐⭐ **Risk**: Medium

**Motivation**: EqProp converges 1.4x faster but 5.2% lower accuracy.

**Proposed Architecture**:
- Use EqProp for attention layers (local credit)
- Use Backprop for feedforward (global optimization)

**Expected**: 20-30% speedup, <2% accuracy loss

## Actionable Quick Wins

### ✅ Reduce EqProp LR by 10x
**Impact**: -15% failure rate  
**Effort**: Update constraints dict  
**Confidence**: High (30% of failures = LR too high)

## Research Gaps
1. No experiments on graph data → Test GNNs
2. Missing adversarial robustness tests
3. No continual learning experiments
```

---

## Phase 6: Infrastructure & Documentation (Week 5-6)

**Goal**: Professional tooling for researchers

### 6.1 Experiment Tracking Integration

**Files to Create**:
- `bioplausible/tracking.py`

```python
import wandb

class ExperimentTracker:
    """Unified experiment tracking."""
    
    def __init__(self, project="bioplausible", backend="wandb"):
        self.backend = backend
        if backend == "wandb":
            wandb.init(project=project)
    
    def log_hyperparams(self, config: Dict):
        wandb.config.update(config)
    
    def log_checkpoint(self, checkpoint: TrainingCheckpoint, step: int):
        wandb.log({
            "train_acc": checkpoint.train_acc,
            "val_acc": checkpoint.val_acc,
            "grad_norm": checkpoint.grad_norm_mean,
            "train_val_gap": checkpoint.train_val_gap,
        }, step=step)
```

### 6.2 Automated Visualization

```python
class ResultVisualizer:
    """Publication-quality plots."""
    
    def plot_training_dynamics_report(self, trajectories: List[TrainingTrajectory]):
        """Generate all standard plots for report."""
        self.plot_learning_curves(trajectories)
        self.plot_convergence_comparison(trajectories)
        self.plot_overfitting_analysis(trajectories)
        self.plot_sample_efficiency_comparison(trajectories)
```

### 6.3 Documentation

**From TODO.md - still relevant**:
- [ ] Add LICENSE file (MIT)
- [ ] Create CHANGELOG.md
- [ ] Add CONTRIBUTING.md
- [ ] MkDocs setup with API reference
- [ ] Tutorials (Beginner, Intermediate, Advanced)

### 6.4 CI/CD Enhancements

**From TODO.md**:
- [ ] Enhanced GitHub Actions (code quality, test matrix, coverage)
- [ ] Docker improvements (multi-stage, GPU support)
- [ ] Automated validation on PR

---

## Implementation Timeline

### Week 1: Critical Stability
- Day 1-2: SafetyWrapper + gradient clipping
- Day 2-3: Algorithm-specific constraints
- Day 3-4: FailureTracker implementation
- Day 4-5: Testing on high-failure models

### Week 2: Task Quality
- Day 1-2: New RL tasks (Pendulum, Acrobot)
- Day 2-3: CharNGram tasks
- Day 3-4: Promotion thresholds (task-specific)
- Day 4-5: Curriculum manager

### Week 3: Training Dynamics
- Day 1-2: TrainingCheckpoint/Trajectory dataclasses
- Day 2-3: Database schema update
- Day 3-4: Checkpoint integration in runner
- Day 4-5: Convergence/overfitting detection

### Week 4: Modular Reporting
- Day 1-2: Report directory structure + ReportComposer
- Day 2-3: Section templates (summary, leaderboards, dynamics)
- Day 3-4: AI-optimized composition
- Day 4-5: Integration with ScientistReporter

### Week 5: AI Synthesis
- Day 1-2: Enhanced ResearchSynthesizer
- Day 2-3: Failure analysis + curriculum suggestions
- Day 3-4: Architecture recommendations
- Day 4-5: Narrative generation

### Week 6: Infrastructure
- Day 1-2: Experiment tracking (W&B)
- Day 2-3: Visualization toolkit
- Day 3-4: Documentation (LICENSE, CHANGELOG, CONTRIBUTING)
- Day 4-5: CI/CD enhancements

---

## Success Metrics

**After Phase 1-2 (Stability)**:
- ✅ Failure rate: 30%+ → <10%
- ✅ Models reaching Standard tier: 0 → 5+
- ✅ CartPole replaced with functional RL task

**After Phase 3-4 (Reporting)**:
- ✅ Training dynamics captured for all trials
- ✅ Modular reports generated
- ✅ AI can answer research questions from reports alone

**After Phase 5-6 (Synthesis & Infrastructure)**:
- ✅ ≥5 actionable recommendations generated per run
- ✅ Experiment tracking integrated
- ✅ Documentation complete (LICENSE, CHANGELOG, tutorials)

---

## Risk Mitigation

1. **LR too conservative** → Monitor convergence time, adjust if >2x slower
2. **New tasks still broken** → Test with baseline first, verify >random
3. **Storage bloat** → Compress checkpoints, archive old reports (keep <5GB)
4. **Synthesis quality low** → Validate recommendations with domain expert

---

## What This Replaces

This TODO3.md **consolidates and supersedes**:

1. **TODO.md** → Infrastructure goals (Phases 6: tracking, viz, docs, CI/CD)
2. **TODO2.md** → Reporting architecture (Phases 3-5: dynamics, modular reports, synthesis)
3. **Report analysis** → Stability fixes (Phases 1-2: safety, constraints, tasks)

**After implementation**, delete TODO.md and TODO2.md.

---

## Quick Reference

### Essential Commands
```bash
# Run with safety enabled
bioplausible-scientist run --safety-mode strict

# Analyze failures
bioplausible-scientist failures --recent 24h

# Generate full report (modular)
bioplausible-scientist report --format modular --output reports/

# Health check
bioplausible-scientist health
```

### Key Files
- `bioplausible/scientist/core.py` - Main scientist logic
- `bioplausible/scientist/safety.py` - Training safety
- `bioplausible/scientist/curriculum.py` - Curriculum manager
- `bioplausible/scientist/training_dynamics.py` - Checkpoint system
- `bioplausible/scientist/report/composer_v2.py` - Modular reports
- `bioplausible/scientist/synthesizer.py` - AI synthesis

---

**Last Updated**: 2026-02-05  
**Version**: Scientist++ v2.0  
**Status**: Ready for implementation
