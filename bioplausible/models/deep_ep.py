"""
Directed Equilibrium Propagation (DEEP)

Implements Equilibrium Propagation with asymmetric forward and feedback weights.
Based on research into relaxing the symmetry constraint (e.g. standard EqProp requires W = W^T).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("directed_ep")
class DirectedEP(BioModel):
    """
    Directed EqProp (DEEP) with separate forward and feedback weights.
    Both sets of weights are updated to minimize the energy/loss.
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Shortcuts from config
        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        self.forward_layers = nn.ModuleList()
        self.feedback_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            # Forward: dim[i] -> dim[i+1]
            fwd = nn.Linear(dims[i], dims[i + 1])
            self.forward_layers.append(fwd)

            # Feedback: dim[i+1] -> dim[i]
            # Initialize feedback weights independently
            bwd = nn.Linear(
                dims[i + 1], dims[i], bias=False
            )  # Bias usually in forward layer
            self.feedback_layers.append(bwd)

        self.to(kwargs.get("device", "cpu"))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward_dynamics(
        self,
        activations: List[torch.Tensor],
        beta: float = 0.0,
        target: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Run one pass of relaxation dynamics using separate feedback weights.
        """
        new_activations = [activations[0]]  # Input is clamped
        num_layers = len(self.forward_layers)

        for i in range(num_layers):
            fwd_layer = self.forward_layers[i]
            h_prev = activations[i]

            # Bottom-up input
            a_bu = fwd_layer(h_prev)

            # Top-down contribution using Explicit Feedback Weights
            a_td = 0.0
            if i < num_layers - 1:
                bwd_layer = self.feedback_layers[i + 1]  # Feedback from layer i+1 to i
                # Note: layer i connects h_i to h_{i+1}.
                # h_{i+1} sends feedback to h_i.
                # forward_layers[i] maps h_i -> h_{i+1}.
                # feedback_layers[i] maps h_{i+1} -> h_i.
                pass

        # Dynamics loop updates activations[1] ... activations[-1].
        # h_0 is fixed.

        updated_activations = [activations[0]]

        for k in range(len(self.forward_layers)):
            # Update h_{k+1}
            # Inputs:
            # - Bottom-up from h_k via W_k
            # - Top-down from h_{k+2} via B_{k+1} (if exists)

            h_prev = activations[k]  # h_k

            # Bottom-up
            a_bu = self.forward_layers[k](h_prev)

            # Top-down
            a_td = 0.0
            if k < len(self.forward_layers) - 1:
                h_next = activations[k + 2]  # h_{k+2}
                # Feedback from k+2 to k+1
                # forward_layers[k+1] goes k+1 -> k+2
                # feedback_layers[k+1] goes k+2 -> k+1
                a_td = self.feedback_layers[k + 1](h_next)

            total = a_bu + a_td

            if k < len(self.forward_layers) - 1:
                h_new = self.activation(total)
            else:
                h_new = total

            if k == len(self.forward_layers) - 1 and beta > 0 and target is not None:
                h_new = h_new + beta * (target - h_new)

            updated_activations.append(h_new)

        return updated_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        eq_steps = steps if steps is not None else self.eq_steps

        activations = [x]
        h = x
        for i, layer in enumerate(self.forward_layers):
            h = layer(h)
            if i < len(self.forward_layers) - 1:
                h = self.activation(h)
            activations.append(h)

        # Storage for dynamics
        trajectory = []
        deltas = []

        if return_trajectory:
            trajectory.append([a.detach().cpu() for a in activations])

        for _ in range(eq_steps):
            prev_activations = activations
            activations = self.forward_dynamics(activations, beta, target)

            if return_dynamics:
                # Calculate change in hidden state
                delta = 0.0
                # activations[0] is input (fixed), so skip
                for k in range(1, len(activations)):
                    delta += (activations[k] - prev_activations[k]).norm().item()
                deltas.append(delta)

            if return_trajectory:
                trajectory.append([a.detach().cpu() for a in activations])

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
        """Update both forward and feedback weights."""
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)

        with torch.no_grad():
            self.forward(x, beta=0.0)
            free = self._last_activations

        with torch.no_grad():
            self.forward(x, beta=self.beta, target=target)
            nudged = self._last_activations

        self.optimizer.zero_grad()

        with torch.no_grad():
            for i in range(len(self.forward_layers)):
                h_prev_free, h_post_free = free[i], free[i + 1]
                h_prev_nudge, h_post_nudge = nudged[i], nudged[i + 1]

                # 1. Update Forward Weights W
                # Standard EqProp rule: dW ~ (h_post h_prev^T)_nudged - ...
                prod_nudged = torch.matmul(h_post_nudge.T, h_prev_nudge)
                prod_free = torch.matmul(h_post_free.T, h_prev_free)

                dW = (prod_nudged - prod_free) / self.beta
                dW /= x.size(0)

                if self.forward_layers[i].weight.grad is None:
                    self.forward_layers[i].weight.grad = -dW
                else:
                    self.forward_layers[i].weight.grad += -dW

                if self.forward_layers[i].bias is not None:
                    db = (h_post_nudge - h_post_free).sum(0) / self.beta
                    db /= x.size(0)
                    if self.forward_layers[i].bias.grad is None:
                        self.forward_layers[i].bias.grad = -db
                    else:
                        self.forward_layers[i].bias.grad += -db

                # 2. Update Feedback Weights B (The "Directed" part)
                # In DEEP, B is also updated.
                # Various rules exist. One is simply updating B with the SAME rule as W
                # (transpose of update), encouraging symmetry (Kolen-Pollack).
                # Another is to update B to minimize reconstruction error (autoencoder-like).
                # Or just use the same contrastive Hebbian rule.

                # We'll use the contrastive rule for B as well.
                # B maps h_post -> h_prev.
                # So dB ~ (h_prev h_post^T)_nudged - ...
                # This is exactly dW^T.

                prod_nudged_b = torch.matmul(h_prev_nudge.T, h_post_nudge)
                prod_free_b = torch.matmul(h_prev_free.T, h_post_free)

                dB = (prod_nudged_b - prod_free_b) / self.beta
                dB /= x.size(0)

                if self.feedback_layers[i].weight.grad is None:
                    self.feedback_layers[i].weight.grad = -dB
                else:
                    self.feedback_layers[i].weight.grad += -dB

        self.optimizer.step()

        loss = nn.functional.cross_entropy(free[-1], y).item()
        acc = (free[-1].argmax(dim=1) == y).float().mean().item()

        return {"loss": loss, "accuracy": acc}
