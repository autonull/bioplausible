from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_energy(model: nn.Module, x: torch.Tensor, h: torch.Tensor) -> float:
    """
    Compute total energy (Hopfield-like) of the system:
    E = 0.5 * ||h - f(h, x)||^2

    In equilibrium, E -> 0.
    """
    with torch.no_grad():
        x_proj = model.W_in(x)
        # Assuming h was the input to the step that produced the NEXT h
        # But for energy of state h, we check consistency with next step implied by h
        next_h_pre = x_proj + model.W_rec(h)
        next_h = torch.tanh(next_h_pre)

        # Energy is magnitude of update vector (proxy for violation of equilibrium)
        # E = 0.5 * || h - tanh(Wx + Wh) ||^2
        diff = h - next_h
        energy = 0.5 * torch.sum(diff**2).item()

    return energy


def estimate_lyapunov(model: nn.Module, x: torch.Tensor, steps: int = 50) -> float:
    """
    Estimate max Lyapunov exponent to detect chaos/stability.

    Method: Track divergence of two infinitesimally close trajectories.
    λ = (1/t) * ln( ||δ(t)|| / ||δ(0)|| )
    """
    batch_size = x.shape[0]

    # Internal check of L
    if hasattr(model, "W_rec"):
        w = model.W_rec.weight
        L = torch.linalg.norm(w, ord=2).item()
        # print(f"    [Inside LE] Internal L={L:.4f}")

    # 1. Trajectory A
    hA = torch.zeros(batch_size, model.hidden_dim, device=x.device)

    # 2. Trajectory B (perturbed)
    epsilon = 1e-3
    # Initialize perturbation with exact norm epsilon
    pert = torch.randn_like(hA)
    pert = pert / torch.norm(pert) * epsilon
    hB = hA + pert

    x_proj = model.W_in(x)

    divergences = []

    for t in range(steps):
        # Update both
        hA = torch.tanh(x_proj + model.W_rec(hA))
        hB = torch.tanh(x_proj + model.W_rec(hB))

        # Measure distance
        dist = torch.norm(hA - hB).item()
        if dist < 1e-9:
            dist = 1e-9  # Avoid log(0)

        # Re-normalize hB to stay close (avoid saturation effects), keeping direction
        # This is standard wolf algorithm / rescaling method for LE
        hB = hA + (hB - hA) / dist * epsilon

        # Log divergence rate
        # Local expansion = dist / epsilon
        if dist == 0:
            divergences.append(-10.0)  # Collapsed
        else:
            ratio = dist / epsilon
            # if t < 5 and ratio > 1.05:
            #    print(f"    Step {t}: ratio={ratio:.4f} (Expansion!)")
            divergences.append(np.log(ratio))

    # Mean Lyapunov Exponent
    # Exclude initial transient steps (burn-in)
    lambda_max = (
        np.mean(divergences[10:]) if len(divergences) > 10 else np.mean(divergences)
    )
    return lambda_max


def analyze_angle_evolution(
    model_over_time: List[nn.Module], reference_grads: List[torch.Tensor]
) -> Dict[str, List[float]]:
    """
    Track how alignment angles evolve during training.
    """
    angles = []
    # This requires capturing model snapshots, which might be heavy.
    # Alternative: compute online during training loop.
    pass


class EnergyMonitor:
    """Stateful monitor for energy landscape visualization."""

    def __init__(self):
        self.energies = []

    def record(self, energy: float):
        self.energies.append(energy)

    def get_plot_ascii(self, height=10) -> str:
        """Simple ASCII plot of energy relaxation."""
        if not self.energies:
            return ""

        vals = np.array(self.energies)
        # Normalize to 0-height
        min_v, max_v = vals.min(), vals.max()
        if max_v == min_v:
            return "-" * len(vals)

        norm = (vals - min_v) / (max_v - min_v)
        rows = []
        for h in range(height, -1, -1):
            row = []
            thresh = h / height
            for v in norm:
                char = "█" if v >= thresh else " "
                row.append(char)
            rows.append("".join(row))

        return "\n".join(rows)
