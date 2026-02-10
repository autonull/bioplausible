"""
AdaptiveFeedbackAlignment - Novel Hybrid Algorithm

FA with slowly-evolving feedback matrix that adapts toward better alignment.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("adaptive_feedback_alignment")
class AdaptiveFeedbackAlignment(BioModel):
    """FA with slow adaptive feedback evolution."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Build default layers if not done by subclass custom logic
        if not hasattr(self, "layers") or len(self.layers) == 0:
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

        # Feedback weights as ParameterList
        self.feedback_weights = nn.ParameterList()
        # Use self.config instead of config, or ensure config is populated
        if config is None:
            config = self.config

        hidden_dims = (
            config.hidden_dims
            if config.hidden_dims
            else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
        )
        dims = [config.input_dim] + hidden_dims + [config.output_dim]

        for i in range(len(dims) - 1):
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(nn.Parameter(B, requires_grad=True))

        self.criterion = nn.CrossEntropyLoss()

        self.w_optimizer = torch.optim.Adam(
            self.layers.parameters(), lr=self.config.learning_rate
        )
        self.b_optimizer = torch.optim.Adam(
            self.feedback_weights.parameters(), lr=self.config.learning_rate * 0.001
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.w_optimizer.zero_grad()
        self.b_optimizer.zero_grad()

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        output = activations[-1]
        loss = self.criterion(output, y)

        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()

        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                h_prev = activations[i]

                if i == len(self.layers) - 1:
                    grad_h = error
                else:
                    grad_h = torch.mm(error, self.feedback_weights[i + 1])
                    h_curr = activations[i + 1]

                    if isinstance(self.activation, nn.ReLU):
                        grad_h = grad_h * (h_curr > 0).float()
                    elif isinstance(self.activation, nn.Tanh):
                        grad_h = grad_h * (1 - h_curr**2)

                grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)

                # Update gradients for W optimizer
                if self.layers[i].weight.grad is None:
                    self.layers[i].weight.grad = grad_W
                else:
                    self.layers[i].weight.grad += grad_W

                if self.layers[i].bias is not None:
                    grad_b = grad_h.mean(0)
                    if self.layers[i].bias.grad is None:
                        self.layers[i].bias.grad = grad_b
                    else:
                        self.layers[i].bias.grad += grad_b

                # Update B to match W
                if i < len(self.layers) - 1:
                    target_B = self.layers[i + 1].weight.data
                    current_B = self.feedback_weights[i + 1].data

                    # This update is non-gradient based (it's a tracking update)
                    # We can keep doing it manually or wrap it in a custom optimizer step?
                    # The original code did:
                    # self.feedback_weights[i+1].data += self.config.learning_rate * 0.001 * diff
                    # But it initialized b_optimizer which was UNUSED.

                    # We can use the optimizer if we define a loss for B.
                    # Loss_B = 0.5 * ||B - W||^2. Grad_B = (B - W).
                    # Update: B = B - lr * (B - W) = B + lr * (W - B).
                    # Matches the logic.

                    grad_B = -(target_B - current_B)
                    if self.feedback_weights[i + 1].grad is None:
                        self.feedback_weights[i + 1].grad = grad_B
                    else:
                        self.feedback_weights[i + 1].grad += grad_B

                error = grad_h

        self.w_optimizer.step()
        self.b_optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }

    @classmethod
    def build(
        cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs
    ):
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            extra=kwargs,
        )
        return cls(config=config).to(device)
