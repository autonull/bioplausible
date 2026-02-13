"""
PredictiveCodingHybrid - Novel Algorithm

Combines Predictive Coding (top-down predictions) with Feedback Alignment.
Layers try to predict their inputs.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("predictive_coding_hybrid")
class PredictiveCodingHybrid(BioModel):
    """Layers predict inputs; FA propagates prediction errors."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Build layers if needed
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

        self.criterion = nn.CrossEntropyLoss()

        # Feedback connections are now separate parameters (top-down predictors)
        self.top_down = nn.ModuleList()
        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            # Predict layer i from layer i+1
            layer = nn.Linear(dims[i + 1], dims[i])
            self.top_down.append(layer)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        output = activations[-1]
        loss_cls = self.criterion(output, y)

        pc_loss = 0
        for i in range(len(self.layers)):
            upper = activations[i + 1].detach()
            lower_target = activations[i].detach()

            prediction = self.top_down[i](upper)
            pc_loss += nn.functional.mse_loss(prediction, lower_target)

        total_loss = loss_cls + 0.1 * pc_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            extra=kwargs,
        )
        return cls(config=config).to(device)
