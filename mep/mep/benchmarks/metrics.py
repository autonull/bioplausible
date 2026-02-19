"""
MEP Benchmark Metrics

Provides MetricsTracker class for collecting and aggregating training metrics.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class MetricsTracker:
    """
    Track and aggregate metrics during training.

    Collects per-step metrics and computes epoch-level averages.
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, List] = defaultdict(list)
        self.epoch_start_time: Optional[float] = None

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()

    def end_epoch(self) -> None:
        """Mark the end of an epoch and record duration."""
        if self.epoch_start_time is not None:
            duration = time.time() - self.epoch_start_time
            self.metrics['epoch_time'].append(duration)
            self.epoch_start_time = None

    def log_step(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        spectral_norm: Optional[float] = None,
        energy_free: Optional[float] = None,
        energy_nudged: Optional[float] = None,
        settling_steps: Optional[int] = None,
        grad_norm: Optional[float] = None
    ) -> None:
        """
        Log metrics for a single training step.

        Args:
            loss: Training loss value
            accuracy: Training accuracy (optional)
            spectral_norm: Spectral norm of weights (optional)
            energy_free: Energy in free phase (optional)
            energy_nudged: Energy in nudged phase (optional)
            settling_steps: Number of settling steps taken (optional)
            grad_norm: Gradient norm (optional)
        """
        self.metrics['step_loss'].append(loss)
        if accuracy is not None:
            self.metrics['step_accuracy'].append(accuracy)
        if spectral_norm is not None:
            self.metrics['step_spectral_norm'].append(spectral_norm)
        if energy_free is not None:
            self.metrics['step_energy_free'].append(energy_free)
        if energy_nudged is not None:
            self.metrics['step_energy_nudged'].append(energy_nudged)
        if settling_steps is not None:
            self.metrics['step_settling_steps'].append(settling_steps)
        if grad_norm is not None:
            self.metrics['step_grad_norm'].append(grad_norm)

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """
        Compute average metrics for the epoch.

        Returns:
            Dictionary mapping metric names to epoch-averaged values.
        """
        epoch_metrics: Dict[str, float] = {}

        for key, values in list(self.metrics.items()):
            if key.startswith('step_') and values:
                epoch_key = key.replace('step_', 'epoch_')
                epoch_metrics[epoch_key] = float(np.mean(values))
                # Clear step metrics to avoid unbounded growth
                self.metrics[key] = []

        # Add epoch time if available
        if 'epoch_time' in self.metrics and self.metrics['epoch_time']:
            epoch_metrics['epoch_time'] = float(self.metrics['epoch_time'][-1])

        return epoch_metrics

    @staticmethod
    def check_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """
        Check if tensor contains NaN or Inf values.

        Args:
            tensor: Tensor to check
            name: Name for error reporting

        Returns:
            True if tensor contains NaN or Inf, False otherwise.
        """
        return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())
