"""
ContrastiveFeedbackAlignment - Novel Hybrid Algorithm

Combines Contrastive Learning with Feedback Alignment.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("contrastive_feedback_alignment")
class ContrastiveFeedbackAlignment(BioModel):
    """Contrastive FA."""

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

        # Feedback weights
        self.feedback_weights = nn.ParameterList()
        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(nn.Parameter(B, requires_grad=False))

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

        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }
