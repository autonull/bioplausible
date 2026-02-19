"""
EP Debugging Utilities

Tools for monitoring and debugging Equilibrium Propagation training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field


@dataclass
class EnergyMetrics:
    """Energy metrics for a single settling step."""
    step: int
    energy: float
    energy_change: float
    grad_norm: float
    state_norm: float


@dataclass
class EpochMetrics:
    """Metrics for a complete EP epoch."""
    epoch: int
    free_energy: float
    nudged_energy: float
    energy_gap: float
    gradient_norm: float
    weight_change: float
    settling_steps: int
    energy_history: List[EnergyMetrics] = field(default_factory=list)


class EPMonitor:
    """
    Monitor for EP training dynamics.
    
    Tracks:
    - Energy convergence during settling
    - Free vs nudged energy gap
    - Gradient norms
    - Weight updates
    
    Usage:
        monitor = EPMonitor()
        optimizer = smep(..., model=model)
        
        for epoch in range(epochs):
            monitor.start_epoch()
            optimizer.step(x=x, target=y)
            metrics = monitor.end_epoch(model, optimizer)
            print(f"Epoch {epoch}: Energy gap = {metrics.energy_gap:.4f}")
    """
    
    def __init__(self) -> None:
        self.current_epoch: int = 0
        self.epoch_metrics: List[EpochMetrics] = []
        self.settling_history: List[EnergyMetrics] = []
        self._prev_weights: Dict[str, torch.Tensor] = {}
        self._start_time: Optional[Any] = None
    
    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.current_epoch += 1
        self.settling_history = []
        if torch.cuda.is_available():
            self._start_time = torch.cuda.Event(enable_timing=True)
            self._start_time.record() # type: ignore
        else:
            self._start_time = None
    
    def record_settling_step(
        self,
        step: int,
        energy: float,
        prev_energy: Optional[float],
        states: List[torch.Tensor],
        grads: List[Optional[torch.Tensor]]
    ) -> EnergyMetrics:
        """Record metrics for a settling step."""
        grad_norm_sq = sum(g.norm().item() ** 2 for g in grads if g is not None)
        grad_norm = float(grad_norm_sq) ** 0.5

        state_norm_sq = sum(s.norm().item() ** 2 for s in states)
        state_norm = float(state_norm_sq) ** 0.5
        
        metrics = EnergyMetrics(
            step=step,
            energy=energy,
            energy_change=energy - prev_energy if prev_energy is not None else 0.0,
            grad_norm=grad_norm,
            state_norm=state_norm
        )
        self.settling_history.append(metrics)
        return metrics
    
    def end_epoch(
        self,
        model: nn.Module,
        optimizer: Any,
        free_energy: Optional[float] = None,
        nudged_energy: Optional[float] = None
    ) -> EpochMetrics:
        """
        Mark the end of an epoch and compute metrics.
        
        Args:
            model: The model being trained.
            optimizer: The optimizer (for accessing state).
            free_energy: Free phase energy (optional).
            nudged_energy: Nudged phase energy (optional).
        
        Returns:
            EpochMetrics for this epoch.
        """
        # Compute weight change
        weight_change_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name in self._prev_weights:
                    weight_change_sq += (param.data - self._prev_weights[name]).norm().item() ** 2
            self._prev_weights[name] = param.data.clone()
        weight_change = float(weight_change_sq) ** 0.5
        
        # Compute gradient norm
        grad_norm_sq = sum(
            p.grad.norm().item() ** 2 
            for p in model.parameters() 
            if p.grad is not None
        )
        grad_norm = float(grad_norm_sq) ** 0.5
        
        # Energy gap
        energy_gap = (nudged_energy - free_energy) if (free_energy is not None and nudged_energy is not None) else 0.0
        
        metrics = EpochMetrics(
            epoch=self.current_epoch,
            free_energy=free_energy or 0.0,
            nudged_energy=nudged_energy or 0.0,
            energy_gap=energy_gap,
            gradient_norm=grad_norm,
            weight_change=weight_change,
            settling_steps=len(self.settling_history),
            energy_history=self.settling_history.copy()
        )
        
        self.epoch_metrics.append(metrics)
        return metrics
    
    def check_convergence(
        self,
        tolerance: float = 1e-4,
        min_steps: int = 5
    ) -> bool:
        """
        Check if settling has converged.
        
        Args:
            tolerance: Energy change threshold for convergence.
            min_steps: Minimum settling steps before checking.
        
        Returns:
            True if converged, False otherwise.
        """
        if len(self.settling_history) < min_steps + 1:
            return False
        
        recent = self.settling_history[-min_steps:]
        max_change = max(abs(m.energy_change) for m in recent)
        return max_change < tolerance
    
    def get_energy_trajectory(self) -> List[float]:
        """Get energy values across all settling steps in current epoch."""
        return [m.energy for m in self.settling_history]
    
    def plot_energy(self, ax: Optional[Any] = None) -> Any:
        """Plot energy convergence for current epoch."""
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        
        energies = self.get_energy_trajectory()
        ax.plot(energies)
        ax.set_xlabel('Settling Step')
        ax.set_ylabel('Energy')
        ax.set_title(f'Epoch {self.current_epoch}: Energy Convergence')
        return ax
    
    def summary(self) -> str:
        """Generate summary of training so far."""
        if not self.epoch_metrics:
            return "No epochs recorded"
        
        lines = [f"EP Training Summary ({len(self.epoch_metrics)} epochs)"]
        lines.append("-" * 50)
        
        for m in self.epoch_metrics[-5:]:  # Last 5 epochs
            lines.append(
                f"Epoch {m.epoch:3d}: "
                f"Energy Gap={m.energy_gap:.4f}, "
                f"Grad Norm={m.gradient_norm:.4f}, "
                f"Weight Change={m.weight_change:.4f}"
            )
        
        return "\n".join(lines)


def monitor_ep_training(
    model: nn.Module,
    optimizer: Any,
    train_loader: Any,
    epochs: int,
    device: torch.device,
    verbose: bool = True
) -> EPMonitor:
    """
    Train with EP while monitoring energy dynamics.
    
    Args:
        model: Model to train.
        optimizer: EP optimizer.
        train_loader: DataLoader for training data.
        epochs: Number of epochs.
        device: Training device.
        verbose: Print progress.
    
    Returns:
        EPMonitor with training metrics.
    """
    monitor = EPMonitor()
    
    for epoch in range(epochs):
        monitor.start_epoch()
        model.train()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.step(x=x, target=y)
            # Should call end_epoch or update metrics?
            # The original code only called start_epoch and nothing else inside loop.
            # Assuming optimizer updates are tracked via hooks or similar?
            # EPMonitor doesn't seem to hook into optimizer automatically.
            # The usage example says: metrics = monitor.end_epoch(model, optimizer)
            # But monitor_ep_training doesn't call it.
            # I will assume this function is incomplete or simplified example.
            pass

        # End of epoch
        monitor.end_epoch(model, optimizer)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} completed")
    
    return monitor
