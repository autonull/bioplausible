"""
StochasticFA - Novel Algorithm

Randomly drops out feedback connections.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("stochastic_fa")
class StochasticFA(BioModel):
    """FA with dropout on feedback signals."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

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

        self.feedback_weights = []
        dims = (
            [self.input_dim]
            + (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
            )
            + [self.output_dim]
        )
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(B)

        self.criterion = nn.CrossEntropyLoss()
        self.drop_prob = 0.5

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate
        )
        self.optimizer.zero_grad()

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

        for i in reversed(range(len(self.layers))):
            h_prev = activations[i]
            if i == len(self.layers) - 1:
                grad_h = error
            else:
                B = self.feedback_weights[i + 1].to(error.device)
                mask = (torch.rand_like(B) > self.drop_prob).float()
                B_effective = B * mask * (1.0 / (1.0 - self.drop_prob))

                grad_h = torch.mm(error, B_effective)
                h_curr = activations[i + 1]
                if isinstance(self.activation, nn.ReLU):
                    grad_h = grad_h * (h_curr > 0).float()

            grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
            self.layers[i].weight.data -= self.config.learning_rate * grad_W
            if self.layers[i].bias is not None:
                self.layers[i].bias.data -= self.config.learning_rate * grad_h.mean(0)
            error = grad_h

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
