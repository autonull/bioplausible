"""
Homeostatic EqProp: Self-Tuning Dynamic Lipschitz Scaling (Track 8)

Implements "Autonomic Homeostasis" - a network that monitors its stability
and automatically adjusts weight scales to maintain L < 1.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import estimate_lipschitz


@dataclass
class HomeostasisMetrics:
    avg_velocity: float
    lipschitz_estimate: float
    brake_applied: float
    boost_applied: float
    layers_braked: int
    layers_boosted: int


class HomeostaticEqProp(nn.Module):
    """
    EqProp with Dynamic Lipschitz Scaling for autonomous stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 5,
        alpha: float = 0.5,
        target_lipschitz: float = 0.95,
        velocity_threshold_high: float = 0.1,
        velocity_threshold_low: float = 0.01,
        adaptation_rate: float = 0.01,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha

        # Regulation parameters
        self.target_lipschitz = target_lipschitz
        self.velocity_threshold_high = velocity_threshold_high
        self.velocity_threshold_low = velocity_threshold_low
        self.adaptation_rate = adaptation_rate

        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Learnable scaling factors (adjusted by homeostasis)
        self.register_buffer("layer_scales", torch.ones(num_layers))

        self.head = nn.Linear(hidden_dim, output_dim)

        # Conservative init
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(0.7)

        self.last_velocities: Dict[int, float] = {}
        self.homeostasis_history: List[HomeostasisMetrics] = []

    def _estimate_layer_lipschitz(self, layer_idx: int) -> float:
        """Estimate effective Lipschitz constant of a layer."""
        # Temporarily scale weight
        original_weight = self.layers[layer_idx].weight
        scaled_weight = original_weight * self.layer_scales[layer_idx]
        # if layer_idx == 0:
        #    print(f"DEBUG: Layer 0 scale={self.layer_scales[layer_idx].item():.4f} W_norm={torch.norm(scaled_weight).item():.4f}")

        # Create a temporary container to use shared utility
        # Wrapper to match utils interface if needed, but simple linear suffices
        with torch.no_grad():
            W = scaled_weight
            u = torch.randn(W.shape[1], device=W.device)
            u = F.normalize(u, dim=0)
            for _ in range(3):
                v = F.normalize(W @ u, dim=0)
                u = F.normalize(W.T @ v, dim=0)
            sigma = torch.norm(W @ u).item()
        return sigma

    def forward_step(
        self,
        h_states: Dict[int, torch.Tensor],
        x: torch.Tensor,
        track_velocity: bool = False,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
        """Single equilibrium step."""
        new_states = {}
        velocities = {}
        x_emb = self.W_in(x)

        for i, layer in enumerate(self.layers):
            pre = x_emb if i == 0 else h_states.get(i - 1, torch.zeros_like(x_emb))
            h_curr = h_states.get(i, torch.zeros_like(pre))

            # Apply scaling
            scale = self.layer_scales[i]
            h_target = torch.tanh(F.linear(pre, layer.weight * scale, layer.bias))

            h_new = (1 - self.alpha) * h_curr + self.alpha * h_target
            new_states[i] = h_new

            if track_velocity:
                velocity = torch.mean(torch.abs(h_new - h_curr)).item()
                velocities[i] = velocity

        return new_states, velocities

    def apply_homeostasis(self, velocities: Dict[int, float]) -> HomeostasisMetrics:
        """
        Apply homeostatic regulation using Proportional Control (P-controller).

        Instead of bang-bang thresholds, we scale weights continuously based on
        how far the velocity/Lipschitz estimates are from targets.
        """
        brake_total = 0.0
        boost_total = 0.0
        layers_braked = 0
        layers_boosted = 0

        for i, velocity in velocities.items():
            # Estimate current stability (L) proxy from velocity
            # High velocity -> High L -> Unstable

            # P-Controller Logic:
            # error = velocity - target
            # scale_change = -k * error

            # If velocity is high (chaos), we shrink.
            # If velocity is low (vanishing), we grow.

            # We use a non-linear mapping to be safe:
            current_L = self._estimate_layer_lipschitz(i)

            # Braking condition: High velocity OR High Lipschitz
            if velocity > self.velocity_threshold_high or current_L > (
                self.target_lipschitz + 0.1
            ):
                # Braking (Shrink)
                error_v = max(0, velocity - self.velocity_threshold_high)
                error_l = max(0, current_L - self.target_lipschitz)

                # Combined error signal
                error = error_v + error_l

                # Stronger response for larger errors
                factor = 1.0 - (self.adaptation_rate * (1.0 + 10.0 * error))
                factor = max(0.5, factor)  # Safety clamp

                self.layer_scales[i] *= factor
                # if i == 0:
                #     print(f"DEBUG: Braking layer {i}. Vel={velocity:.5f} L={current_L:.4f} Factor={factor:.4f} NewScale={self.layer_scales[i].item():.4f}")
                brake_total += 1.0 - factor
                layers_braked += 1

            elif velocity < self.velocity_threshold_low:
                # Boosting (Expand) - but check Lipschitz guardrail
                current_L = self._estimate_layer_lipschitz(i)
                if current_L < self.target_lipschitz:
                    error = self.velocity_threshold_low - velocity
                    factor = 1.0 + (self.adaptation_rate * (1.0 + 5.0 * error))
                    factor = min(1.5, factor)  # Safety clamp

                    self.layer_scales[i] *= factor
                    boost_total += factor - 1.0
                    layers_boosted += 1

        # Hard limits on scaling to prevent total collapse or explosion
        self.layer_scales.clamp_(0.1, 3.0)

        avg_velocity = sum(velocities.values()) / len(velocities) if velocities else 0.0
        avg_lipschitz = (
            sum(self._estimate_layer_lipschitz(i) for i in range(self.num_layers))
            / self.num_layers
        )

        metrics = HomeostasisMetrics(
            avg_velocity=avg_velocity,
            lipschitz_estimate=avg_lipschitz,
            brake_applied=brake_total,
            boost_applied=boost_total,
            layers_braked=layers_braked,
            layers_boosted=layers_boosted,
        )

        self.homeostasis_history.append(metrics)
        self.last_velocities = velocities

        return metrics

    def forward(
        self, x: torch.Tensor, steps: int = 30, apply_homeostasis: bool = True
    ) -> torch.Tensor:
        """Forward pass with auto-regulation."""
        batch_size = x.size(0)
        h_states = {
            i: torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for i in range(self.num_layers)
        }

        all_velocities = []
        for step in range(steps):
            track = step >= steps // 2
            h_states, velocities = self.forward_step(h_states, x, track_velocity=track)
            if track:
                all_velocities.append(velocities)

        if apply_homeostasis and all_velocities:
            avg_velocities = {}
            for i in range(self.num_layers):
                avg_velocities[i] = sum(v.get(i, 0) for v in all_velocities) / len(
                    all_velocities
                )
            self.apply_homeostasis(avg_velocities)

        return self.head(h_states[self.num_layers - 1])

    def get_stability_report(self) -> str:
        """Generate stability status report."""
        lipschitz = [self._estimate_layer_lipschitz(i) for i in range(self.num_layers)]
        max_L = max(lipschitz) if lipschitz else 0.0
        status = "✓ STABLE" if max_L < 1.0 else "⚠ UNSTABLE"

        lines = [
            f"Max Lipschitz: {max_L:.4f} {status}",
            f"Layer Scales: {[f'{s:.3f}' for s in self.layer_scales.tolist()]}",
        ]
        if self.homeostasis_history:
            last = self.homeostasis_history[-1]
            lines.append(
                f"Last Action: {last.layers_braked} braked, {last.layers_boosted} boosted"
            )

        return "\n".join(lines)
