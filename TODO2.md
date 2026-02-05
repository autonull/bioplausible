# TODO2: Scientist++ - Next-Generation Autonomous Research System

> **Status**: Design Complete - Ready for Implementation  
> **Priority**: High - Unlocks 10x richer experimental insights  
> **Effort**: 4-6 weeks (phased implementation)

---

## Executive Summary

This document defines **Scientist++**, an evolution of the current Scientist system that transforms it from a hyperparameter optimization tool into a **comprehensive autonomous research platform**. The key improvements are:

1. **Algorithm-Aware Hyperparameter Metamodel**: Intelligent constraints that understand which hyperparameters apply to which algorithm families
2. **Continuous Training Dynamics**: Replace discrete tiers with progressive checkpoints capturing full learning trajectories
3. **Modular Report Architecture**: Multi-file reports enabling AI analysis and flexible composition
4. **Research Synthesis Engine**: Automated generation of cross-algorithm insights and actionable recommendations

---

## Part 1: Algorithm-Aware Hyperparameter Metamodel

### 1.1 Problem Statement

**Current Issue**: The search space treats all hyperparameters as universal:
```python
# Current approach (naive)
EQPROP_PARAMS = {
    "lr": (1e-5, 1e-2, "log"),
    "beta": (0.05, 0.5, "linear"),
    # Problem: What about optimizer? activation? These aren't universal!
}
```

**Key Insight**: Different algorithm families have fundamentally different parameter spaces:
- **Backprop**: Uses optimizers (Adam, SGD, etc.), dropout, standard activations
- **EqProp**: Uses `beta`, `steps`, energy-based dynamics (no gradient descent optimizer)
- **Feedback Alignment**: Uses `fa_scale`, random feedback matrices
- **Hebbian**: Uses contrastive phases, local learning rules

### 1.2 Proposed Solution: Hyperparameter Taxonomy

Create a **three-tier hyperparameter system**:

```python
# bioplausible/hyperopt/hyperparameter_metamodel.py

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

class HyperparamScope(Enum):
    """Defines which algorithms a hyperparameter applies to."""
    UNIVERSAL = "universal"        # All algorithms (lr, hidden_dim, etc.)
    GRADIENT_BASED = "gradient"    # Backprop, variants (optimizer, grad_clip)
    EQUILIBRIUM = "equilibrium"    # EqProp family (beta, steps, nudge_type)
    FEEDBACK_ALIGNMENT = "fa"      # FA variants (fa_scale, adapt_rate)
    HEBBIAN = "hebbian"            # CHL, etc. (contrastive_steps)
    TRANSFORMER = "transformer"    # Transformer-specific (num_heads, etc.)

@dataclass
class HyperparamSpec:
    """Specification for a single hyperparameter."""
    name: str
    scope: HyperparamScope
    param_type: str  # "continuous", "discrete", "categorical"
    
    # For continuous/discrete
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    scale: Optional[str] = None  # "log", "linear", "int"
    
    # For categorical
    choices: Optional[List[Any]] = None
    
    # Conditional dependencies
    requires: Optional[List[str]] = None  # Other hyperparams that must exist
    conflicts: Optional[List[str]] = None  # Hyperparams that cannot coexist
    
    # Metadata
    description: str = ""
    default: Any = None

# Universal hyperparameters (apply to ALL algorithms)
UNIVERSAL_HYPERPARAMS = [
    HyperparamSpec(
        name="lr",
        scope=HyperparamScope.UNIVERSAL,
        param_type="continuous",
        range_min=1e-5,
        range_max=1e-1,
        scale="log",
        description="Learning rate for weight updates",
        default=1e-3,
    ),
    HyperparamSpec(
        name="hidden_dim",
        scope=HyperparamScope.UNIVERSAL,
        param_type="discrete",
        choices=[32, 64, 128, 256, 512],
        description="Number of hidden units per layer",
        default=128,
    ),
    HyperparamSpec(
        name="num_layers",
        scope=HyperparamScope.UNIVERSAL,
        param_type="discrete",
        range_min=1,
        range_max=30,
        scale="int",
        description="Number of layers in the network",
        default=4,
    ),
    HyperparamSpec(
        name="activation",
        scope=HyperparamScope.UNIVERSAL,
        param_type="categorical",
        choices=["relu", "gelu", "silu", "tanh", "leaky_relu", "elu"],
        description="Activation function (NOTE: some algorithms constrain this)",
        default="silu",
    ),
    HyperparamSpec(
        name="weight_init",
        scope=HyperparamScope.UNIVERSAL,
        param_type="categorical",
        choices=["xavier", "kaiming", "orthogonal", "lecun"],
        description="Weight initialization scheme",
        default="kaiming",
    ),
]

# Gradient-based only (Backprop and gradient-based variants)
GRADIENT_HYPERPARAMS = [
    HyperparamSpec(
        name="optimizer",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="categorical",
        choices=["sgd", "adam", "adamw", "rmsprop"],
        description="Gradient descent optimizer (ONLY for Backprop)",
        default="adam",
    ),
    HyperparamSpec(
        name="weight_decay",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=0.0,
        range_max=1e-2,
        scale="log",
        description="L2 regularization strength",
        default=0.0,
    ),
    HyperparamSpec(
        name="grad_clip",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=0.0,
        range_max=10.0,
        scale="linear",
        description="Gradient clipping threshold",
        default=1.0,
    ),
    HyperparamSpec(
        name="dropout",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=0.0,
        range_max=0.5,
        scale="linear",
        description="Dropout probability",
        default=0.0,
    ),
    HyperparamSpec(
        name="momentum",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=0.0,
        range_max=0.99,
        scale="linear",
        description="SGD momentum (only if optimizer=sgd)",
        requires=["optimizer"],  # Conditional
        default=0.9,
    ),
]

# Equilibrium Propagation family
EQUILIBRIUM_HYPERPARAMS = [
    HyperparamSpec(
        name="beta",
        scope=HyperparamScope.EQUILIBRIUM,
        param_type="continuous",
        range_min=0.01,
        range_max=1.0,
        scale="linear",
        description="Nudge strength for clamping (EqProp only)",
        default=0.1,
    ),
    HyperparamSpec(
        name="steps",
        scope=HyperparamScope.EQUILIBRIUM,
        param_type="discrete",
        range_min=5,
        range_max=50,
        scale="int",
        description="Number of relaxation steps (EqProp only)",
        default=20,
    ),
    HyperparamSpec(
        name="nudge_type",
        scope=HyperparamScope.EQUILIBRIUM,
        param_type="categorical",
        choices=["output_clamping", "energy_based", "symmetric"],
        description="How to apply target nudging",
        default="output_clamping",
    ),
]

# Feedback Alignment family
FA_HYPERPARAMS = [
    HyperparamSpec(
        name="fa_scale",
        scope=HyperparamScope.FEEDBACK_ALIGNMENT,
        param_type="continuous",
        range_min=0.5,
        range_max=2.0,
        scale="linear",
        description="Scaling factor for feedback weights",
        default=1.0,
    ),
    HyperparamSpec(
        name="adapt_rate",
        scope=HyperparamScope.FEEDBACK_ALIGNMENT,
        param_type="continuous",
        range_min=1e-4,
        range_max=1e-1,
        scale="log",
        description="Adaptation rate for feedback weights",
        default=1e-2,
    ),
]

# Hebbian family
HEBBIAN_HYPERPARAMS = [
    HyperparamSpec(
        name="contrastive_steps",
        scope=HyperparamScope.HEBBIAN,
        param_type="discrete",
        range_min=5,
        range_max=30,
        scale="int",
        description="Steps in contrastive phase",
        default=10,
    ),
]

# Transformer-specific
TRANSFORMER_HYPERPARAMS = [
    HyperparamSpec(
        name="num_heads",
        scope=HyperparamScope.TRANSFORMER,
        param_type="categorical",
        choices=[2, 4, 8],
        description="Number of attention heads",
        default=4,
    ),
    HyperparamSpec(
        name="context_length",
        scope=HyperparamScope.TRANSFORMER,
        param_type="discrete",
        choices=[64, 128, 256, 512],
        description="Maximum sequence length",
        default=128,
    ),
]

class HyperparameterMetamodel:
    """
    Central registry that knows which hyperparameters apply to which algorithms.
    """
    
    def __init__(self):
        self.all_specs = (
            UNIVERSAL_HYPERPARAMS +
            GRADIENT_HYPERPARAMS +
            EQUILIBRIUM_HYPERPARAMS +
            FA_HYPERPARAMS +
            HEBBIAN_HYPERPARAMS +
            TRANSFORMER_HYPERPARAMS
        )
        self._spec_dict = {spec.name: spec for spec in self.all_specs}
    
    def get_search_space_for_model(self, model_spec: 'ModelSpec') -> Dict[str, HyperparamSpec]:
        """
        Return the appropriate hyperparameters for a given model.
        
        Uses the model's family to determine which scoped params apply.
        """
        applicable_scopes = {HyperparamScope.UNIVERSAL}
        
        # Map model families to hyperparameter scopes
        family = model_spec.family.lower()
        
        if family == "baseline":
            # Backprop uses gradient-based hyperparams
            applicable_scopes.add(HyperparamScope.GRADIENT_BASED)
        
        elif family == "eqprop":
            # EqProp uses equilibrium hyperparams, NO optimizer
            applicable_scopes.add(HyperparamScope.EQUILIBRIUM)
        
        elif family == "hybrid":
            # Hybrid models (e.g., Adaptive FA) might use both
            # Need to check model_type for specifics
            if "fa" in model_spec.model_type or "alignment" in model_spec.model_type:
                applicable_scopes.add(HyperparamScope.FEEDBACK_ALIGNMENT)
            if "equilibrium" in model_spec.model_type or "eq" in model_spec.model_type:
                applicable_scopes.add(HyperparamScope.EQUILIBRIUM)
            # Hybrids might also use gradient methods
            applicable_scopes.add(HyperparamScope.GRADIENT_BASED)
        
        elif family == "hebbian":
            applicable_scopes.add(HyperparamScope.HEBBIAN)
        
        # Transformers get transformer-specific params
        if "transformer" in model_spec.model_type:
            applicable_scopes.add(HyperparamScope.TRANSFORMER)
            # Transformers also use gradient-based training (usually)
            if family != "eqprop":  # Unless it's EqProp Transformer
                applicable_scopes.add(HyperparamScope.GRADIENT_BASED)
            else:
                applicable_scopes.add(HyperparamScope.EQUILIBRIUM)
        
        # Filter specs by applicable scopes
        search_space = {}
        for spec in self.all_specs:
            if spec.scope in applicable_scopes:
                search_space[spec.name] = spec
        
        # Apply algorithm-specific activation constraints
        # Example: Holomorphic EqProp REQUIRES tanh (holomorphic)
        if model_spec.name == "Holomorphic EqProp":
            search_space["activation"].choices = ["tanh"]
        
        # Example: Spiking networks might require step activations
        # (not implemented yet, but shows extensibility)
        
        return search_space
    
    def validate_config(self, model_spec: 'ModelSpec', config: Dict[str, Any]) -> List[str]:
        """
        Validate that a config is compatible with a model.
        Returns list of error messages (empty if valid).
        """
        errors = []
        valid_space = self.get_search_space_for_model(model_spec)
        
        for key, value in config.items():
            if key not in valid_space:
                errors.append(
                    f"Hyperparameter '{key}' is not applicable to {model_spec.name} "
                    f"(family: {model_spec.family})"
                )
        
        # Check for missing required params
        for key, spec in valid_space.items():
            if spec.requires:
                for req in spec.requires:
                    if req not in config:
                        errors.append(
                            f"Hyperparameter '{key}' requires '{req}' but it's missing"
                        )
        
        return errors

# Global instance
HYPERPARAM_METAMODEL = HyperparameterMetamodel()
```

### 1.3 Usage in Scientist

```python
# bioplausible/scientist/core.py (updated)

from bioplausible.hyperopt.hyperparameter_metamodel import HYPERPARAM_METAMODEL
from bioplausible.models.registry import get_model_spec

def create_optuna_trial(trial, model_name: str):
    """Create trial configuration using metamodel."""
    model_spec = get_model_spec(model_name)
    search_space = HYPERPARAM_METAMODEL.get_search_space_for_model(model_spec)
    
    config = {}
    for param_name, spec in search_space.items():
        if spec.param_type == "continuous":
            config[param_name] = trial.suggest_float(
                param_name, spec.range_min, spec.range_max, log=(spec.scale == "log")
            )
        elif spec.param_type == "discrete":
            if spec.choices:
                config[param_name] = trial.suggest_categorical(param_name, spec.choices)
            else:
                config[param_name] = trial.suggest_int(
                    param_name, int(spec.range_min), int(spec.range_max)
                )
        elif spec.param_type == "categorical":
            config[param_name] = trial.suggest_categorical(param_name, spec.choices)
    
    # Validate before returning
    errors = HYPERPARAM_METAMODEL.validate_config(model_spec, config)
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    
    return config
```

**Benefit**: Optuna will now:
- Suggest `optimizer` for Backprop but NOT for EqProp
- Suggest `beta` for EqProp but NOT for Backprop
- Automatically handle algorithm-specific constraints

---

## Part 2: Continuous Training Dynamics

### 2.1 Progressive Checkpoint System

**Replace discrete tiers** (2, 5, 30, 100 epochs) with **continuous checkpoints**:

```python
# bioplausible/scientist/training_dynamics.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class TrainingCheckpoint:
    """Metrics captured at a single checkpoint."""
    epoch: int
    train_acc: float
    val_acc: float
    test_acc: Optional[float]  # Only computed at final checkpoint
    train_loss: float
    val_loss: float
    
    # Training dynamics (NEW!)
    grad_norm_mean: float
    grad_norm_std: float
    weight_norm: float
    learning_rate: float  # May decay over time
    
    # Overfitting indicators
    train_val_gap: float  # train_acc - val_acc
    
    # Task-specific metrics
    perplexity: Optional[float] = None  # For LM tasks
    reward: Optional[float] = None      # For RL tasks
    
    # Efficiency metrics
    wall_time_seconds: float = 0.0
    total_flops: Optional[int] = None

@dataclass
class TrainingTrajectory:
    """Complete training history for one trial."""
    trial_id: int
    model_name: str
    task_name: str
    config: dict
    checkpoints: List[TrainingCheckpoint]
    
    # Derived metrics (computed from checkpoints)
    convergence_epoch: Optional[int] = None  # Epoch where improvement plateaus
    converged: bool = False
    overfitting_detected: bool = False
    unstable: bool = False  # Large loss variance
    
    def compute_convergence_speed(self) -> float:
        """
        Epochs to reach 90% of final accuracy.
        Lower is faster.
        """
        if not self.checkpoints:
            return float('inf')
        
        final_acc = self.checkpoints[-1].val_acc
        target_acc = 0.9 * final_acc
        
        for i, ckpt in enumerate(self.checkpoints):
            if ckpt.val_acc >= target_acc:
                return ckpt.epoch
        
        return float('inf')  # Never converged
    
    def compute_sample_efficiency(self) -> float:
        """
        Area under the learning curve (AUC).
        Higher is better (learns faster with fewer samples).
        """
        epochs = [c.epoch for c in self.checkpoints]
        accs = [c.val_acc for c in self.checkpoints]
        return np.trapz(accs, epochs) / epochs[-1] if epochs else 0.0
    
    def detect_overfitting(self, threshold: float = 0.1) -> bool:
        """
        Returns True if train-val gap exceeds threshold.
        """
        if len(self.checkpoints) < 3:
            return False
        
        # Check last 3 checkpoints
        recent_gaps = [c.train_val_gap for c in self.checkpoints[-3:]]
        return any(gap > threshold for gap in recent_gaps)

class ContinuousTrainingSchedule:
    """
    Manages progressive training with adaptive checkpointing.
    """
    
    # Standard checkpoints (can be overridden)
    DEFAULT_CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200]
    
    def __init__(self, max_budget: int = 100, enable_pruning: bool = True):
        self.max_budget = max_budget
        self.enable_pruning = enable_pruning
        self.checkpoints = [c for c in self.DEFAULT_CHECKPOINTS if c <= max_budget]
    
    def train_with_checkpoints(
        self, 
        model, 
        task, 
        config: dict,
        trial  # Optuna trial for pruning
    ) -> TrainingTrajectory:
        """
        Train model with periodic evaluation at checkpoints.
        """
        trajectory = TrainingTrajectory(
            trial_id=trial.number,
            model_name=model.name,
            task_name=task.name,
            config=config,
            checkpoints=[],
        )
        
        last_epoch = 0
        for target_epoch in self.checkpoints:
            # Train from last_epoch to target_epoch
            metrics = self._train_epochs(
                model, task, last_epoch, target_epoch
            )
            
            # Create checkpoint
            ckpt = TrainingCheckpoint(
                epoch=target_epoch,
                train_acc=metrics["train_acc"],
                val_acc=metrics["val_acc"],
                train_loss=metrics["train_loss"],
                val_loss=metrics["val_loss"],
                grad_norm_mean=metrics["grad_norm_mean"],
                grad_norm_std=metrics["grad_norm_std"],
                weight_norm=metrics["weight_norm"],
                learning_rate=config["lr"],  # May adjust for decay
                train_val_gap=metrics["train_acc"] - metrics["val_acc"],
                perplexity=metrics.get("perplexity"),
                wall_time_seconds=metrics["wall_time"],
            )
            trajectory.checkpoints.append(ckpt)
            
            # Report to Optuna for pruning
            trial.report(ckpt.val_acc, target_epoch)
            
            # Check if should prune
            if self.enable_pruning and trial.should_prune():
                trajectory.converged = False
                break
            
            last_epoch = target_epoch
        
        # Compute derived metrics
        trajectory.convergence_epoch = self._find_convergence(trajectory)
        trajectory.overfitting_detected = trajectory.detect_overfitting()
        trajectory.unstable = self._check_stability(trajectory)
        
        return trajectory
    
    def _train_epochs(self, model, task, start_epoch, end_epoch):
        """Train for specified epoch range and return metrics."""
        # Implementation depends on model/task
        # This is a placeholder
        pass
    
    def _find_convergence(self, trajectory: TrainingTrajectory) -> Optional[int]:
        """
        Detect convergence point (where improvement plateaus).
        Uses sliding window to check if accuracy stops improving.
        """
        if len(trajectory.checkpoints) < 3:
            return None
        
        window_size = 3
        improvement_threshold = 0.01  # 1% improvement required
        
        for i in range(len(trajectory.checkpoints) - window_size):
            window = trajectory.checkpoints[i:i+window_size]
            improvement = window[-1].val_acc - window[0].val_acc
            
            if improvement < improvement_threshold:
                return window[0].epoch
        
        return None  # Still improving
    
    def _check_stability(self, trajectory: TrainingTrajectory) -> bool:
        """Check if training is unstable (high loss variance)."""
        if len(trajectory.checkpoints) < 5:
            return False
        
        recent_losses = [c.train_loss for c in trajectory.checkpoints[-5:]]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Consider unstable if coefficient of variation > 0.5
        return (loss_std / loss_mean) > 0.5 if loss_mean > 0 else False
```

### 2.2 Database Schema Extension

Add new table for training trajectories:

```sql
-- bioplausible/hyperopt/schema_v2.sql

CREATE TABLE IF NOT EXISTS training_trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    task_name TEXT NOT NULL,
    config JSON NOT NULL,
    convergence_epoch INTEGER,
    converged BOOLEAN,
    overfitting_detected BOOLEAN,
    unstable BOOLEAN,
    FOREIGN KEY (trial_id) REFERENCES trials(id)
);

CREATE TABLE IF NOT EXISTS training_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_acc REAL,
    val_acc REAL,
    test_acc REAL,
    train_loss REAL,
    val_loss REAL,
    grad_norm_mean REAL,
    grad_norm_std REAL,
    weight_norm REAL,
    learning_rate REAL,
    train_val_gap REAL,
    perplexity REAL,
    reward REAL,
    wall_time_seconds REAL,
    total_flops INTEGER,
    FOREIGN KEY (trajectory_id) REFERENCES training_trajectories(id)
);

CREATE INDEX idx_checkpoints_trajectory ON training_checkpoints(trajectory_id);
CREATE INDEX idx_checkpoints_epoch ON training_checkpoints(epoch);
```

**Storage Impact**: 
- Current: ~500KB per trial (final metrics only)
- New: ~2MB per trial (8-10 checkpoints × ~200KB each)
- For 1000 trials: ~2GB total (acceptable)

---

## Part 3: Modular Report Architecture

### 3.1 Problem: Monolithic index.md

**Current**: Single 25KB `index.md` file with everything mixed together
- Hard for AI to parse specific sections
- Can't compose partial reports (e.g., "only MNIST results")
- Difficult to update incrementally

### 3.2 Solution: Multi-File Report Structure

```
reports/
  2026-02-05_10-30-00/
    manifest.json              # Report metadata + file index
    
    # Core sections (always generated)
    01_summary.md              # Executive summary
    02_discoveries.md          # Key findings (auto-generated)
    03_leaderboards.md         # Task-by-task rankings
    04_training_dynamics.md    # Learning curves, convergence analysis
    05_hyperparameter_analysis.md  # Decision trees, feature importance
    06_statistical_tests.md    # Significance matrices, p-values
    
    # Per-task deep dives
    tasks/
      mnist_analysis.md
      cifar10_analysis.md
      tiny_shakespeare_analysis.md
      cartpole_analysis.md
    
    # Per-algorithm deep dives
    algorithms/
      eqprop_mlp_analysis.md   # Detailed analysis for each algorithm
      backprop_baseline_analysis.md
      ...
    
    # Detailed experiment cards
    experiments/
      trial_0042.md            # Individual trial details
      trial_0078.md
      ...
    
    # AI-consumable data
    metadata.json              # Structured data for AI analysis
    
    # Images (as before)
    images/
      leaderboard_mnist.png
      curves_eqprop_family.png
      ...
    
    # Compilation targets (generated by combining files)
    FULL_REPORT.md            # Concatenation of all sections
    MNIST_ONLY.md             # Only MNIST-related sections
    AI_ANALYSIS_READY.md      # Subset optimized for LLM input
```

### 3.3 Manifest Format

```json
{
  "report_version": "2.0",
  "generated_at": "2026-02-05T10:30:00Z",
  "total_experiments": 236,
  "models_tested": 20,
  "tasks": ["mnist", "cifar10", "tiny_shakespeare", "cartpole"],
  
  "sections": [
    {
      "file": "01_summary.md",
      "title": "Executive Summary",
      "type": "summary",
      "dependencies": []
    },
    {
      "file": "02_discoveries.md",
      "title": "Key Discoveries",
      "type": "insights",
      "dependencies": ["metadata.json"]
    },
    {
      "file": "04_training_dynamics.md",
      "title": "Training Dynamics Analysis",
      "type": "dynamics",
      "dependencies": ["images/curves_*.png"]
    }
  ],
  
  "composite_reports": {
    "FULL_REPORT": ["01_summary.md", "02_discoveries.md", "03_leaderboards.md", ...],
    "MNIST_ONLY": ["01_summary.md", "tasks/mnist_analysis.md", "algorithms/eqprop_mlp_analysis.md"],
    "AI_ANALYSIS": ["02_discoveries.md", "metadata.json", "05_hyperparameter_analysis.md"]
  },
  
  "best_models": {
    "mnist": {"model": "EqProp MLP", "accuracy": 0.9031, "trial_id": 42},
    "cifar10": {...},
    ...
  }
}
```

### 3.4 Report Composition Utility

```python
# bioplausible/scientist/report_composer.py

import json
from pathlib import Path
from typing import List

class ReportComposer:
    """Compose modular reports into various formats."""
    
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.manifest = self._load_manifest()
    
    def _load_manifest(self):
        with open(self.report_dir / "manifest.json") as f:
            return json.load(f)
    
    def compose_full_report(self) -> str:
        """Generate FULL_REPORT.md by concatenating all sections."""
        sections = self.manifest["composite_reports"]["FULL_REPORT"]
        return self._concatenate_files(sections)
    
    def compose_task_report(self, task_name: str) -> str:
        """Generate report for a specific task."""
        files = [
            "01_summary.md",
            f"tasks/{task_name}_analysis.md",
            "03_leaderboards.md",  # Will filter to this task
        ]
        return self._concatenate_files(files)
    
    def compose_ai_analysis_input(self) -> str:
        """
        Generate AI-optimized report.
        Includes structured data + narrative.
        """
        sections = self.manifest["composite_reports"]["AI_ANALYSIS"]
        
        # Prepend metadata as markdown code block
        metadata_str = json.dumps(self.manifest, indent=2)
        preamble = f"""# AI Analysis Dataset

## Metadata
```json
{metadata_str}
```

## Analysis Sections

"""
        return preamble + self._concatenate_files(sections)
    
    def _concatenate_files(self, file_list: List[str]) -> str:
        """Concatenate markdown files with section separators."""
        output = []
        for file_path in file_list:
            full_path = self.report_dir / file_path
            if full_path.exists():
                output.append(f"\n\n---\n\n")
                output.append(full_path.read_text())
            else:
                output.append(f"\n\n> **Note**: Section `{file_path}` not found.\n\n")
        
        return "".join(output)
    
    def export_for_ai(self, output_path: Path):
        """Export AI-ready JSON + Markdown bundle."""
        ai_report = {
            "metadata": self.manifest,
            "narrative": self.compose_ai_analysis_input(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(ai_report, f, indent=2)

# Usage:
# composer = ReportComposer(Path("reports/2026-02-05_10-30-00"))
# composer.compose_full_report()  # -> FULL_REPORT.md
# composer.compose_task_report("mnist")  # -> MNIST_ONLY.md
```

---

## Part 4: AI-Powered Research Synthesis

### 4.1 Synthesis Engine Architecture

```python
# bioplausible/scientist/synthesizer.py

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@dataclass
class CrossAlgorithmInsight:
    """A comparative insight between algorithm families."""
    insight_type: str  # "performance", "efficiency", "robustness"
    task: str
    metric: str
    ranking: List[str]  # Algorithm families ranked by performance
    effect_sizes: Dict[str, float]  # Cohen's d effect sizes
    confidence: float  # Statistical confidence (0-1)
    narrative: str  # Human-readable description

@dataclass
class ArchitecturalRecommendation:
    """A proposed novel architecture based on experiment results."""
    name: str
    motivation: str
    architecture_description: str
    expected_benefits: List[str]
    implementation_sketch: str
    risk_level: str  # "low", "medium", "high"
    priority: int  # 1-5, higher is more urgent

class ResearchSynthesizer:
    """
    Generates high-level research insights from experiment database.
    """
    
    def __init__(self, trajectories: List[TrainingTrajectory]):
        self.trajectories = trajectories
        self.algorithm_families = self._group_by_family()
    
    def synthesize_full_report(self) -> Dict[str, Any]:
        """Main entry point for synthesis."""
        return {
            "cross_algorithm_insights": self.generate_cross_algorithm_insights(),
            "architectural_recommendations": self.generate_architecture_recommendations(),
            "research_gaps": self.identify_research_gaps(),
            "actionable_quick_wins": self.find_quick_wins(),
        }
    
    def generate_cross_algorithm_insights(self) -> List[CrossAlgorithmInsight]:
        """
        Compare algorithm families across tasks and metrics.
        """
        insights = []
        
        for task in ["mnist", "cifar10", "tiny_shakespeare", "cartpole"]:
            # Compare final accuracy
            insights.append(self._compare_families(
                task=task,
                metric="final_accuracy",
                insight_type="performance"
            ))
            
            # Compare sample efficiency (AUC of learning curve)
            insights.append(self._compare_families(
                task=task,
                metric="sample_efficiency",
                insight_type="efficiency"
            ))
            
            # Compare convergence speed
            insights.append(self._compare_families(
                task=task,
                metric="convergence_speed",
                insight_type="efficiency"
            ))
        
        return insights
    
    def _compare_families(self, task: str, metric: str, insight_type: str) -> CrossAlgorithmInsight:
        """Compare algorithm families on a single metric."""
        # Get all trajectories for this task
        task_trajs = [t for t in self.trajectories if t.task_name == task]
        
        # Group by family
        family_scores = {}
        for family, trajs in self.algorithm_families.items():
            scores = []
            for traj in trajs:
                if traj.task_name == task:
                    if metric == "final_accuracy":
                        scores.append(traj.checkpoints[-1].val_acc)
                    elif metric == "sample_efficiency":
                        scores.append(traj.compute_sample_efficiency())
                    elif metric == "convergence_speed":
                        scores.append(traj.compute_convergence_speed())
            
            if scores:
                family_scores[family] = np.mean(scores)
        
        # Rank families
        ranking = sorted(family_scores.keys(), key=lambda f: family_scores[f], reverse=True)
        
        # Compute effect sizes (Cohen's d)
        effect_sizes = self._compute_effect_sizes(family_scores)
        
        # Generate narrative
        narrative = self._generate_narrative(task, metric, ranking, family_scores)
        
        return CrossAlgorithmInsight(
            insight_type=insight_type,
            task=task,
            metric=metric,
            ranking=ranking,
            effect_sizes=effect_sizes,
            confidence=0.95,  # Placeholder, could compute from p-values
            narrative=narrative,
        )
    
    def generate_architecture_recommendations(self) -> List[ArchitecturalRecommendation]:
        """
        Propose novel architectures based on observed strengths.
        """
        recommendations = []
        
        # Example: If EqProp is fast but less accurate than Backprop
        eqprop_speed = self._avg_metric("eqprop", "convergence_speed")
        backprop_speed = self._avg_metric("baseline", "convergence_speed")
        eqprop_acc = self._avg_metric("eqprop", "final_accuracy")
        backprop_acc = self._avg_metric("baseline", "final_accuracy")
        
        if eqprop_speed < backprop_speed and backprop_acc > eqprop_acc:
            recommendations.append(ArchitecturalRecommendation(
                name="Hybrid EqProp-Backprop Transformer",
                motivation=(
                    f"EqProp converges {backprop_speed/eqprop_speed:.1f}x faster "
                    f"but achieves {(backprop_acc - eqprop_acc)*100:.1f}% lower accuracy. "
                    "Combining them could yield best of both worlds."
                ),
                architecture_description=(
                    "Use EqProp for attention layers (local credit assignment) "
                    "and Backprop for feedforward layers (global optimization)."
                ),
                expected_benefits=[
                    "20-30% faster training",
                    "<2% accuracy loss vs pure Backprop",
                    "Better biological plausibility"
                ],
                implementation_sketch="""
class HybridTransformer(nn.Module):
    def __init__(self):
        self.attention = EqPropAttention(...)  # Use EqProp
        self.ffn = nn.Linear(...)              # Use Backprop
    
    def forward(self, x):
        # Attention with EqProp dynamics
        attn_out = self.attention.equilibrate(x)
        # FFN with standard backprop
        return self.ffn(attn_out)
""",
                risk_level="medium",
                priority=5,
            ))
        
        # Add more recommendations based on other patterns...
        
        return recommendations
    
    def identify_research_gaps(self) -> List[str]:
        """Find under-explored areas."""
        gaps = []
        
        # Check for missing task types
        tested_tasks = set(t.task_name for t in self.trajectories)
        if "graph" not in tested_tasks:
            gaps.append("No experiments on graph neural networks (GNNs)")
        
        # Check for under-explored hyperparameters
        all_configs = [t.config for t in self.trajectories]
        activations_tested = set(c.get("activation") for c in all_configs)
        if "prelu" not in activations_tested:
            gaps.append("PReLU (learnable activation) not tested - may help EqProp")
        
        # Check for missing robustness tests
        # (would need to track this in metadata)
        
        return gaps
    
    def find_quick_wins(self) -> List[Dict[str, Any]]:
        """
        Identify low-hanging fruit: simple changes with big impact.
        """
        wins = []
        
        # Example: If GELU consistently outperforms SiLU
        accuracy_by_activation = self._group_by_hyperparam("activation")
        if "gelu" in accuracy_by_activation and "silu" in accuracy_by_activation:
            gelu_acc = np.mean(accuracy_by_activation["gelu"])
            silu_acc = np.mean(accuracy_by_activation["silu"])
            
            if gelu_acc > silu_acc + 0.03:  # 3% improvement
                wins.append({
                    "title": "Switch default activation to GELU",
                    "impact": f"+{(gelu_acc - silu_acc)*100:.1f}% accuracy",
                    "effort": "1 line code change",
                    "confidence": "high",
                })
        
        return wins
    
    def _group_by_family(self) -> Dict[str, List[TrainingTrajectory]]:
        """Group trajectories by algorithm family."""
        families = {}
        for traj in self.trajectories:
            # Would need to look up family from model registry
            family = "eqprop"  # Placeholder
            families.setdefault(family, []).append(traj)
        return families
    
    def _compute_effect_sizes(self, family_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute Cohen's d effect sizes."""
        # Simplified version
        return {family: score for family, score in family_scores.items()}
    
    def _generate_narrative(self, task, metric, ranking, scores):
        """Generate human-readable description."""
        best = ranking[0]
        worst = ranking[-1]
        return (
            f"On {task}, {best} achieves the best {metric} "
            f"({scores[best]:.3f}), outperforming {worst} ({scores[worst]:.3f})."
        )
    
    def _avg_metric(self, family: str, metric: str) -> float:
        """Compute average metric for a family."""
        # Placeholder
        return 0.85
    
    def _group_by_hyperparam(self, param: str) -> Dict[str, List[float]]:
        """Group accuracy by hyperparameter value."""
        grouped = {}
        for traj in self.trajectories:
            value = traj.config.get(param)
            if value:
                grouped.setdefault(value, []).append(traj.checkpoints[-1].val_acc)
        return grouped
```

### 4.2 Example Synthesis Output

```markdown
# AI-Generated Research Synthesis

## Cross-Algorithm Insights

### Sample Efficiency
**MNIST**: EqProp family (AUC: 0.87) > Feedback Alignment (0.79) > Hebbian (0.71)
**Interpretation**: EqProp learns efficiently from few samples, making it suitable for low-data regimes.

### Convergence Speed
**Tiny Shakespeare**: Feedback Alignment (12 epochs) < Backprop (18 epochs) < EqProp (25 epochs)
**Interpretation**: FA converges fastest on language tasks but with lower final accuracy.

## Architectural Recommendations

### 🔬 Recommendation #1: Hybrid EqProp-Backprop Transformer
**Priority**: ⭐⭐⭐⭐⭐ (High)  
**Risk**: Medium

**Motivation**: EqProp converges 1.4x faster but achieves 5.2% lower accuracy than Backprop.

**Proposed Architecture**:
- Use EqProp for **attention layers** (benefits from local credit assignment)
- Use Backprop for **feedforward layers** (requires global optimization)

**Expected Benefits**:
- 20-30% training speedup
- <2% accuracy loss vs pure Backprop
- Improved biological plausibility

**Next Steps**:
1. Implement in `bioplausible/models/hybrid_transformer.py`
2. Test on WikiText-2 (larger dataset than TinyShakespeare)
3. Benchmark against pure Backprop baseline

---

## Actionable Quick Wins

### ✅ Quick Win #1: Switch to GELU Activation
**Impact**: +3.2% accuracy (averaged across EqProp models)  
**Effort**: 1-line code change  
**Confidence**: High (p < 0.001)

```python
# bioplausible/models/base.py
- activation: str = "silu"
+ activation: str = "gelu"
```

---

## Research Gaps

1. **No experiments on graph data** → Test GNNs with bioplausible learning
2. **Missing adversarial robustness tests** → Add noise injection to CIFAR-10
3. **No continual learning experiments** → Test catastrophic forgetting resistance
```

---

## Part 5: Implementation Roadmap

### Phase 1: Hyperparameter Metamodel (Week 1-2)
- [ ] Implement `HyperparameterMetamodel` class
- [ ] Define all hyperparameter specs (universal, gradient, equilibrium, etc.)
- [ ] Update `get_search_space_for_model()` to use metamodel
- [ ] Add validation logic
- [ ] Write unit tests for metamodel
- [ ] Verify: Run smoke test with Backprop (should suggest `optimizer`) and EqProp (should NOT)

### Phase 2: Continuous Training (Week 2-3)
- [ ] Implement `TrainingCheckpoint` and `TrainingTrajectory` dataclasses
- [ ] Create `ContinuousTrainingSchedule` with progressive checkpoints
- [ ] Update database schema (add tables for trajectories/checkpoints)
- [ ] Modify trial runner to save metrics at each checkpoint
- [ ] Implement early stopping via Optuna pruning
- [ ] Write unit tests for convergence detection
- [ ] Verify: Run 10 trials and confirm checkpoints are saved to DB

### Phase 3: Modular Reporting (Week 3-4)
- [ ] Design report directory structure
- [ ] Implement `ReportComposer` class
- [ ] Create template generators for each section type
- [ ] Generate `manifest.json` with section metadata
- [ ] Implement composition functions (full, per-task, AI-optimized)
- [ ] Write unit tests for report composition
- [ ] Verify: Generate report for existing experiments, manually inspect structure

### Phase 4: AI Synthesis (Week 4-6)
- [ ] Implement `ResearchSynthesizer` class
- [ ] Add cross-algorithm comparison logic
- [ ] Build architecture recommendation engine
- [ ] Create research gap detector
- [ ] Add quick-win identifier
- [ ] Generate narrative summaries
- [ ] Verify: Review 5 generated recommendations for plausibility

### Phase 5: Integration & Testing (Week 6)
- [ ] Integrate all components into `scientist/core.py`
- [ ] Run full end-to-end test (50+ trials across 3 models)
- [ ] Generate complete Scientist++ report
- [ ] Compare with old report format
- [ ] Benchmark compute overhead (\<10% acceptable)
- [ ] User acceptance testing

---

## Backward Compatibility

**All existing code continues to work**:

1. **Old tier system** → Maps to checkpoint budgets:
   ```python
   PatientLevel.SMOKE → max_budget=2, checkpoints=[1, 2]
   PatientLevel.SHALLOW → max_budget=5, checkpoints=[1, 2, 5]
   PatientLevel.STANDARD → max_budget=30, checkpoints=[1, 2, 5, 10, 20, 30]
   ```

2. **Old search spaces** → Converted to metamodel:
   ```python
   SEARCH_SPACES["EqProp MLP"] → HYPERPARAM_METAMODEL.get_search_space_for_model(EqProp)
   ```

3. **Old reports** → Still generated at `index.md`:
   ```python
   # New code generates both:
   generate_legacy_report(out / "index.md")  # Old format
   generate_modular_report(out)              # New format
   ```

---

## Success Metrics

This implementation will be considered successful if:

1. ✅ **Hyperparameter Quality**: Discovers configs 10%+ better in ≥3 tasks
2. ✅ **Compute Efficiency**: Saves 25%+ total compute via early stopping
3. ✅ **Report Completeness**: AI can answer "Which algorithm is best for low-data vision tasks?" from report alone
4. ✅ **Research Insights**: Generates ≥5 actionable recommendations validated by domain expert
5. ✅ **Backward Compat**: \<2% regression on existing benchmarks

---

## Technical Considerations

### Storage
- **Current**: 500KB/trial × 1000 trials = 500MB
- **Proposed**: 2MB/trial × 1000 trials = 2GB
- **Mitigation**: Compress checkpoints, archive old reports

### Compute Overhead
- **Checkpointing**: +5% (periodic evaluation)
- **Early Stopping**: -30% (prune bad trials)
- **Net**: 25% reduction in total compute

### Database Migration
```sql
-- Add new tables (schema_v2.sql)
-- No changes to existing tables (backward compatible)
ALTER TABLE trials ADD COLUMN trajectory_id INTEGER;
```

---

## Future Extensions (Post-v1)

1. **Multi-Fidelity Optimization**: Train on 32×32 images first, then 224×224
2. **Transfer Learning Experiments**: Auto-test MNIST → Fashion-MNIST
3. **Causal Inference**: Use do-calculus to determine *why* hyperparams matter
4. **Interactive Dashboard**: Web UI for exploring training curves
5. **LLM Narrative Generation**: GPT-4 writes full paper drafts from results

---

## Conclusion

**Scientist++** transforms hyperparameter optimization into comprehensive autonomous research. By implementing:

1. **Algorithm-aware hyperparameters** (e.g., optimizer for Backprop only)
2. **Continuous training dynamics** (full learning curves, not just final accuracy)
3. **Modular reports** (composable, AI-analyzable)
4. **Research synthesis** (actionable insights, not just data dumps)

We unlock:
- **Better algorithms** (10%+ accuracy gains from proper hyperparam exploration)
- **Deeper understanding** (training dynamics reveal *why* algorithms work)
- **Faster research** (AI-generated recommendations guide next experiments)

This is not just an optimization tool—it's an **autonomous research collaborator**.
