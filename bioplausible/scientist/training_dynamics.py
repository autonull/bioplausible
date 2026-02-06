import json
import sqlite3
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class TrainingCheckpoint:
    """Metrics captured at a single checkpoint."""
    epoch: int
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float
    
    # Training dynamics
    grad_norm_mean: float = 0.0
    grad_norm_std: float = 0.0
    weight_norm: float = 0.0
    learning_rate: float = 0.0  # May decay over time
    
    # Overfitting indicators
    train_val_gap: float = 0.0  # train_acc - val_acc
    
    # Task-specific metrics
    test_acc: Optional[float] = None  # Only computed at final checkpoint or specific times
    perplexity: Optional[float] = None  # For LM tasks
    reward: Optional[float] = None      # For RL tasks
    
    # Efficiency metrics
    wall_time_seconds: float = 0.0
    total_flops: Optional[int] = None
    samples_seen: int = 0

@dataclass
class TrainingTrajectory:
    """Complete training history for one trial."""
    trial_id: int
    model_name: str
    task_name: str
    config: Dict[str, Any]
    checkpoints: List[TrainingCheckpoint] = field(default_factory=list)
    
    # Derived metrics (computed from checkpoints)
    convergence_epoch: Optional[int] = None  # Epoch where improvement plateaus
    converged: bool = False
    overfitting_detected: bool = False
    unstable: bool = False  # Large loss variance
    
    def compute_convergence_speed(self) -> float:
        """
        Epochs to reach 90% of final accuracy.
        Lower is faster. Returns infinity if no checkpoints or never reached.
        """
        if not self.checkpoints:
            return float('inf')
        
        final_acc = self.checkpoints[-1].val_acc
        target_acc = 0.9 * final_acc
        
        for ckpt in self.checkpoints:
            # We assume checkpoints are sorted by epoch
            if ckpt.val_acc >= target_acc:
                return float(ckpt.epoch)
        
        return float('inf')
    
    def compute_sample_efficiency(self) -> float:
        """
        Area under the learning curve (AUC).
        Higher is better (learns faster with fewer samples).
        """
        if not self.checkpoints:
            return 0.0
            
        epochs = [c.epoch for c in self.checkpoints]
        accs = [c.val_acc for c in self.checkpoints]
        
        # Trapezoidal rule
        # Normalize by total epochs to get average accuracy over time
        if epochs[-1] == 0:
            return 0.0
            
        try:
            # New NumPy 2.0+
            if hasattr(np, "trapezoid"):
                 area = np.trapezoid(accs, epochs)
            # Old NumPy < 2.0
            elif hasattr(np, "trapz"):
                 area = np.trapz(accs, epochs)
            else:
                 raise AttributeError("No trapezoid function")
        except:
            # Manual implementation
            area = 0.0
            for i in range(len(epochs) - 1):
                width = epochs[i+1] - epochs[i]
                height = (accs[i+1] + accs[i]) / 2.0
                area += width * height
                
        return float(area) / epochs[-1]
    
    def detect_overfitting(self, threshold: float = 0.1) -> bool:
        """
        Returns True if train-val gap exceeds threshold.
        """
        if len(self.checkpoints) < 2:
            return False
        
        # Check last 3 checkpoints or fewer if not available
        check_count = min(3, len(self.checkpoints))
        recent_gaps = [c.train_val_gap for c in self.checkpoints[-check_count:]]
        return any(gap > threshold for gap in recent_gaps)

class ContinuousTrainingSchedule:
    """
    Manages progressive training with adaptive checkpointing.
    """
    
    # Standard checkpoints (logarithmic-ish scale)
    DEFAULT_CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
    
    def __init__(self, max_epochs: int = 100, enable_pruning: bool = True):
        self.max_epochs = max_epochs
        self.enable_pruning = enable_pruning
        # Filter checkpoints that are beyond max_epochs
        self.checkpoints = [c for c in self.DEFAULT_CHECKPOINTS if c < max_epochs]
        # Always include the final epoch
        if not self.checkpoints or self.checkpoints[-1] != max_epochs:
            self.checkpoints.append(max_epochs)
    
    def train_with_checkpoints(
        self, 
        trainer,  # The training object (e.g. from bioplausible.training.Trainer)
        trial_id: int,
        model_name: str,
        task_name: str,
        config: Dict[str, Any],
        optuna_trial=None,
        pruning_callback=None,
        on_epoch_end=None
    ) -> TrainingTrajectory:
        """
        Train model with periodic evaluation at checkpoints.
        
        Args:
            trainer: Object with train_epoch() method returning metrics
            trial_id: ID of the trial
            model_name: Name of the model
            task_name: Name of the task
            config: Configuration dict
            optuna_trial: Optional Optuna trial object
            pruning_callback: Optional callback(trial_id, epoch, metrics) -> bool
            on_epoch_end: Optional callback(epoch, metrics) -> None
            
        Returns:
            Completed TrainingTrajectory
        """
        trajectory = TrainingTrajectory(
            trial_id=trial_id,
            model_name=model_name,
            task_name=task_name,
            config=config,
            checkpoints=[],
        )
        
        current_epoch = 0
        cumulative_time = 0.0
        
        # Ensure we start from 0 params if needed, but trainer usually handles init
        
        for target_epoch in self.checkpoints:
            # How many epochs to run in this chunk
            epochs_to_run = target_epoch - current_epoch
            
            if epochs_to_run <= 0:
                continue
                
            chunk_metrics = []
            for _ in range(epochs_to_run):
                # Run one epoch
                m = trainer.train_epoch()
                chunk_metrics.append(m)
                cumulative_time += m.get("time", 0.0)
                current_epoch += 1
                
                if on_epoch_end:
                    on_epoch_end(current_epoch, m)
            
            # Use the metrics from the LAST epoch of this chunk for the checkpoint
            # (or could average them, but snapshot is standard)
            last_metrics = chunk_metrics[-1]
            
            # Compute train_val_gap
            t_acc = last_metrics.get("train_acc", 0.0)
            v_acc = last_metrics.get("accuracy", 0.0) # Standard trainer uses 'accuracy' for validation
            if "val_acc" in last_metrics:
                v_acc = last_metrics["val_acc"]
                
            gap = t_acc - v_acc
            
            # Placeholder for grad norms if not provided
            g_norm = last_metrics.get("grad_norm", 0.0)
            w_norm = last_metrics.get("weight_norm", 0.0)
            
            ckpt = TrainingCheckpoint(
                epoch=target_epoch,
                train_acc=t_acc,
                val_acc=v_acc,
                train_loss=last_metrics.get("train_loss", last_metrics.get("loss", 0.0)),
                val_loss=last_metrics.get("val_loss", last_metrics.get("loss", 0.0)), # Sometimes same if no separate val set
                grad_norm_mean=g_norm, 
                grad_norm_std=0.0, # Would need collection during epoch
                weight_norm=w_norm,
                learning_rate=config.get("lr", 0.0), # Simplification
                train_val_gap=gap,
                perplexity=last_metrics.get("perplexity"),
                reward=last_metrics.get("reward"),
                wall_time_seconds=cumulative_time,
                samples_seen=last_metrics.get("samples_seen", 0)
            )
            
            trajectory.checkpoints.append(ckpt)
            
            # Pruning integration
            if optuna_trial and self.enable_pruning:
                optuna_trial.report(v_acc, target_epoch)
                if optuna_trial.should_prune():
                    trajectory.converged = False
                    print(f"✂️ Trial {trial_id} PRUNED at epoch {target_epoch}")
                    break
            
            elif pruning_callback and self.enable_pruning:
                # Use generic callback if provided
                if pruning_callback(trial_id, target_epoch, last_metrics):
                    trajectory.converged = False
                    print(f"✂️ Trial {trial_id} PRUNED at epoch {target_epoch}")
                    break
        
        # Post-training analysis
        trajectory.convergence_epoch = self._find_convergence(trajectory)
        trajectory.overfitting_detected = trajectory.detect_overfitting()
        trajectory.unstable = self._check_stability(trajectory)
        trajectory.converged = (not trajectory.unstable)
        
        return trajectory

    def _find_convergence(self, trajectory: TrainingTrajectory) -> Optional[int]:
        """
        Detect convergence point (where improvement plateaus).
        """
        if len(trajectory.checkpoints) < 3:
            return None
        
        window_size = 3
        improvement_threshold = 0.01
        
        for i in range(len(trajectory.checkpoints) - window_size + 1):
            window = trajectory.checkpoints[i:i+window_size]
            improvement = window[-1].val_acc - window[0].val_acc
            
            # If improvement over the window is very small
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
        
        # Avoid division by zero
        if loss_mean == 0:
            return False
            
        return (loss_std / loss_mean) > 0.5
