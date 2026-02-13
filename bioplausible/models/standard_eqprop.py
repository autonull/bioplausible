"""
Standard Equilibrium Propagation

Reference implementation with correct top-down feedback dynamics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..acceleration import compile_settling_loop
from .base import BioModel, ModelConfig, register_model


@register_model("eqprop")
class StandardEqProp(BioModel):
    """
    Standard EqProp with free/nudged phases and bidirectional relaxation.

    Implements the dynamics:
    h_i = sigma(W_i h_{i-1} + W_{i+1}^T h_{i+1} + b_i)
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Shortcuts from config
        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

        # Build MLP layers
        self.layers = nn.ModuleList()
        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            layer = self.apply_spectral_norm(layer)
            self.layers.append(layer)

        self.to(kwargs.get("device", "cpu"))

        # Initialize Optimizer
        # Note: We use Adam by default as requested/implied by benchmark
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_spectral_normalized_weight(self, layer: nn.Module) -> torch.Tensor:
        """Get spectral normalized weight, with caching in eval mode."""
        # Check for cached weight in eval mode
        if not self.training and hasattr(layer, "_cached_sn_weight"):
            return layer._cached_sn_weight

        # Compute normalized weight (accessing .weight triggers spectral_norm if present)
        # Note: In PyTorch implementation, accessing .weight on a spectral_norm wrapped layer
        # triggers the recomputation.
        weight = layer.weight

        # Cache in eval mode
        if not self.training:
            layer._cached_sn_weight = weight.detach()

        return weight

    def train(self, mode: bool = True):
        """Override train to clear caches."""
        super().train(mode)
        if mode:  # Entering training mode, clear cache
            for module in self.modules():
                if hasattr(module, "_cached_sn_weight"):
                    delattr(module, "_cached_sn_weight")
        return self

    @compile_settling_loop
    def forward_dynamics(
        self,
        activations: List[torch.Tensor],
        beta: float = 0.0,
        target: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Run one pass of relaxation dynamics over all layers.
        """
        new_activations = [activations[0]]  # Input is clamped

        num_layers = len(self.layers)

        for i in range(num_layers):
            layer = self.layers[i]
            h_prev = activations[i]  # h_{i}

            # OPTIMIZATION: Use cached weight and functional call in eval mode
            if not self.training:
                w = self._get_spectral_normalized_weight(layer)
                b = layer.bias
                a_bu = torch.nn.functional.linear(h_prev, w, b)
            else:
                a_bu = layer(h_prev)

            # Top-down contribution
            a_td = 0.0
            if i < num_layers - 1:
                next_layer = self.layers[i + 1]
                h_next = activations[i + 2]
                if hasattr(next_layer, "weight"):
                    # OPTIMIZATION: Use cached weight for top-down feedback
                    if not self.training:
                        w = self._get_spectral_normalized_weight(next_layer)
                    else:
                        w = next_layer.weight
                    a_td = torch.matmul(h_next, w)

            total_input = a_bu + a_td

            if i < num_layers - 1:
                h_new = self.activation(total_input)
            else:
                h_new = total_input

            if i == num_layers - 1 and beta > 0 and target is not None:
                h_new = h_new + beta * (target - h_new)

            new_activations.append(h_new)

        return new_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Run equilibrium dynamics.
        """
        eq_steps = steps if steps is not None else self.eq_steps

        # Initial feedforward pass
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        # Storage for dynamics
        if return_trajectory:
            # OPTIMIZATION: Preallocate trajectory
            trajectory = [None] * (eq_steps + 1)
            trajectory[0] = [a.detach().cpu() for a in activations]
        else:
            trajectory = None

        deltas = [] if return_dynamics else None

        for step_idx in range(eq_steps):
            prev_activations = activations
            activations = self.forward_dynamics(activations, beta, target)

            # Calculate change in hidden state
            delta = 0.0
            # activations[0] is input (fixed), so skip
            for k in range(1, len(activations)):
                # OPTIMIZATION: Use torch.dist to avoid intermediate allocations (L2 norm)
                delta += torch.dist(activations[k], prev_activations[k], p=2).item()

            if return_dynamics:
                deltas.append(delta)

            # OPTIMIZATION: Adaptive Epsilon Early Stopping
            if step_idx > 5 and delta < 1e-3:
                break

            if return_trajectory:
                trajectory[step_idx + 1] = [a.detach().cpu() for a in activations]

        self._last_activations = activations
        out = activations[-1]

        if return_dynamics:
            return out, {
                "trajectory": trajectory if return_trajectory else None,
                "deltas": deltas,
                "final_delta": deltas[-1] if deltas else 0.0,
            }

        if return_trajectory:
            return out, trajectory

        return out

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """EqProp training step with contrastive phases."""
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)

        # Free phase (beta=0)
        with torch.no_grad():
            self.forward(x, beta=0.0)
            free_activations = self._last_activations
            output_free = free_activations[-1]

        # Nudged phase (beta > 0)
        with torch.no_grad():
            self.forward(x, beta=self.beta, target=target)
            nudged_activations = self._last_activations

        # Contrastive update
        self.optimizer.zero_grad()

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                h_prev_free = free_activations[i]
                h_post_free = free_activations[i + 1]

                h_prev_nudged = nudged_activations[i]
                h_post_nudged = nudged_activations[i + 1]

                prod_nudged = torch.matmul(h_post_nudged.T, h_prev_nudged)
                prod_free = torch.matmul(h_post_free.T, h_prev_free)

                # dW = (prod_nudged - prod_free) / beta
                # This is the negative gradient direction (ascent on energy / descent on loss)
                # Standard EqProp update: W += lr * (nudged - free) / beta
                # Which means grad = -(nudged - free) / beta

                dW = (prod_nudged - prod_free) / self.beta
                dW = dW / x.size(0)

                # Set gradient manually
                param_container = layer
                weight_name = "weight"

                if hasattr(layer, "parametrizations") and hasattr(
                    layer.parametrizations, "weight"
                ):
                    param_container = layer.parametrizations.weight
                    weight_name = "original"

                w_param = getattr(param_container, weight_name)

                # Gradient should be negative of the Hebbian update
                if w_param.grad is None:
                    w_param.grad = -dW
                else:
                    w_param.grad += -dW

                if layer.bias is not None:
                    db = (h_post_nudged - h_post_free).sum(0) / self.beta
                    db = db / x.size(0)
                    if layer.bias.grad is None:
                        layer.bias.grad = -db
                    else:
                        layer.bias.grad += -db

        # Apply optimizer step
        self.optimizer.step()

        # Metrics
        pred = output_free.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        loss = nn.functional.cross_entropy(output_free, y).item()

        return {
            "loss": loss,
            "accuracy": acc,
        }
