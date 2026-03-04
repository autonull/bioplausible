"""
Experiment Tracking Integration

Provides unified interface for logging experiments to Weights & Biases (wandb)
or other backends (MLflow, TensorBoard - future).
"""

import os
import warnings
from typing import Any, Dict, Optional

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Usage:
        tracker = ExperimentTracker(project="bioplausible", name="experiment_1")
        tracker.log_hyperparams({"lr": 0.01})
        for step in range(100):
            tracker.log_metrics({"loss": 0.5}, step=step)
        tracker.finish()
    """

    def __init__(
        self,
        project: str = "bioplausible",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        backend: str = "wandb",
    ):
        self.backend = backend
        self.run = None
        self._step = 0

        if backend == "wandb":
            if not HAS_WANDB:
                warnings.warn(
                    "wandb not installed. Tracking disabled. Install with 'pip install wandb'."
                )
                self.backend = "dummy"
                return

            # Check for API key in env or login
            # We assume user is logged in or env var WANDB_API_KEY is set.
            # If not, wandb.init might prompt or run in offline mode.
            try:
                # Default to disabled if not explicitly requested via env var
                mode = os.environ.get("WANDB_MODE", "disabled")
                self.run = wandb.init(
                    project=project, name=name, config=config, reinit=True, mode=mode
                )
            except Exception as e:
                # Silent failure is preferred if disabled
                if os.environ.get("WANDB_MODE") != "disabled":
                    warnings.warn(
                        f"Failed to initialize wandb: {e}. Tracking disabled."
                    )
                self.backend = "dummy"

        elif backend == "dummy":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def log_hyperparams(self, config: Dict[str, Any]):
        """Log hyperparameters/config."""
        if self.backend == "wandb" and self.run:
            wandb.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics (loss, accuracy, etc.)."""
        if step is None:
            step = self._step
            self._step += 1

        if self.backend == "wandb" and self.run:
            wandb.log(metrics, step=step)

    def log_lipschitz(self, L: float, step: Optional[int] = None):
        """
        Log Lipschitz constant (critical for EqProp stability).
        Logs both the raw value and a boolean 'is_contractive' (L < 1).
        """
        metrics = {"lipschitz_constant": L, "is_contractive": float(L < 1.0)}
        self.log_metrics(metrics, step=step)

    def log_validation_track(self, track_id: int, results: Dict[str, Any]):
        """
        Log results from a specific verification track.
        """
        metrics = {
            f"track_{track_id}/score": results.get("score", 0.0),
            f"track_{track_id}/evidence_level": results.get("evidence_level", 0.0),
            f"track_{track_id}/passed": int(results.get("passed", False)),
        }
        self.log_metrics(metrics)

    def log_image(self, key: str, image_path: str, caption: Optional[str] = None):
        """Log an image artifact."""
        if self.backend == "wandb" and self.run:
            wandb.log({key: wandb.Image(image_path, caption=caption)})

    def log_config(self, cfg: Dict[str, Any]):
        """Log the entire RunConfig dictionary."""
        self.log_hyperparams(cfg)

    def log_energy(self, profile: Any, step: Optional[int] = None):
        """Log an EnergyProfile."""
        metrics = {
            "energy/forward_flops": profile.forward_flops,
            "energy/backward_flops": profile.backward_flops,
            "energy/energy_proxy": profile.energy_proxy,
            "energy/wall_time_ms": profile.wall_time_ms,
            "energy/peak_memory_mb": profile.peak_memory_mb,
            "energy/activation_sparsity": profile.activation_sparsity,
            "energy/weight_sparsity": profile.weight_sparsity,
            "energy/requires_backward": int(profile.requires_backward),
        }
        self.log_metrics(metrics, step=step)

    def finish(self):
        """Close the run."""
        if self.backend == "wandb" and self.run:
            self.run.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
