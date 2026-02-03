"""
Holomorphic Equilibrium Propagation (hEP)

Implements Equilibrium Propagation with complex-valued states and weights.
Based on Laborieux et al. (NeurIPS 2024).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("holomorphic_ep")
class HolomorphicEP(BioModel):
    """
    Holomorphic EqProp with complex-valued weights and states.
    Uses complex tanh activation which is holomorphic.
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Shortcuts from config
        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

        # Build MLP layers with complex weights
        self.layers = nn.ModuleList()
        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            # Custom complex linear layer
            # We use standard nn.Linear but cast params to complex
            layer = nn.Linear(dims[i], dims[i + 1])
            layer.weight = nn.Parameter(layer.weight.to(torch.complex64))
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.to(torch.complex64))

            # Apply spectral norm? Spectral norm for complex matrices exists but
            # PyTorch's spectral_norm might not support complex directly out of box in older versions.
            # We'll skip SN for now or implement a simple one if needed.
            # self.layers.append(self.apply_spectral_norm(layer))
            self.layers.append(layer)

        self.to(kwargs.get("device", "cpu"))

        # We need a complex-aware optimizer, but Adam usually handles complex params in recent PyTorch
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        # Complex tanh is holomorphic
        return torch.tanh(x)

    def forward_dynamics(
        self,
        activations: List[torch.Tensor],
        beta: float = 0.0,
        target: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Run one pass of relaxation dynamics over all layers (Complex domain).
        """
        new_activations = [activations[0]]  # Input is clamped

        num_layers = len(self.layers)

        for i in range(num_layers):
            layer = self.layers[i]
            h_prev = activations[i]  # h_{i}

            # Forward input
            # nn.Linear supports complex input if weights are complex
            a_bu = layer(h_prev)

            # Top-down contribution (Transpose of complex matrix = conjugate transpose?)
            # In EqProp, weights are symmetric W_ij = W_ji.
            # For complex, Hermitian symmetry usually implies W = W^H (conj transpose).
            # So backward weights should be conj(W).T

            a_td = 0.0 + 0.0j
            if i < num_layers - 1:
                next_layer = self.layers[i + 1]
                h_next = activations[i + 2]
                if hasattr(next_layer, "weight"):
                    w = next_layer.weight
                    # W^T * h_next in real case.
                    # W^H * h_next in complex case.
                    # We assume Hermitian symmetry for energy function E = -h^H W h
                    # dE/dh = -W h

                    # We need h_next @ conj(W)
                    # Linear(x) computes x @ W.T
                    # So we need x @ W.H.T = x @ conj(W)

                    w_backward = w.conj().T
                    # Manually compute matmul: [B, H_next] @ [H_next, H_curr]
                    a_td = torch.matmul(
                        h_next, w_backward.T
                    )  # .T because we want (w_backward @ h_next.T).T

            total_input = a_bu + a_td

            if i < num_layers - 1:
                h_new = self.activation(total_input)
            else:
                h_new = total_input

            if i == num_layers - 1 and beta > 0 and target is not None:
                # Nudge output towards target
                # Target is likely real (one-hot), cast to complex
                if not target.is_complex():
                    target = target.to(h_new.dtype)

                h_new = h_new + beta * (target - h_new)

            new_activations.append(h_new)

        return new_activations

    def forward(
        self, x: torch.Tensor, beta: float = 0.0, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Run equilibrium dynamics.
        """
        if not x.is_complex():
            x = x.to(torch.complex64)

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        for _ in range(self.eq_steps):
            activations = self.forward_dynamics(activations, beta, target)

        self._last_activations = activations

        # Return magnitude for classification if needed, or real part?
        # Usually we use the magnitude or real part of logits for classification
        return activations[-1].real

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """EqProp training step with contrastive phases in complex domain."""
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        target = target.to(torch.complex64)  # Ensure complex

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

                # Update rule for complex:
                # dW = (h_post_nudged * h_prev_nudged.H - h_post_free * h_prev_free.H) / beta
                # Note: x.H is conjugate transpose.

                # Batch wise:
                # prod = h_post.T @ h_prev.conj()  <-- if h is [Features, Batch]
                # but h is [Batch, Features].
                # So we want [Features_out, Features_in] matrix.
                # W is [Features_out, Features_in].

                # h_post: [B, Out]
                # h_prev: [B, In]
                # Update: h_post.T @ h_prev.conj()

                prod_nudged = torch.matmul(h_post_nudged.T, h_prev_nudged.conj())
                prod_free = torch.matmul(h_post_free.T, h_prev_free.conj())

                dW = (prod_nudged - prod_free) / self.beta
                dW = dW / x.size(0)

                # Set gradient manually
                if layer.weight.grad is None:
                    layer.weight.grad = -dW
                else:
                    layer.weight.grad += -dW

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
        # Prediction based on real part of output
        pred = output_free.real.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        # Loss for reporting (use real CE on real part of logits)
        loss = nn.functional.cross_entropy(output_free.real, y).item()

        return {
            "loss": loss,
            "accuracy": acc,
        }
