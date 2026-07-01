"""
SparseEquilibrium - Novel Algorithm

Only top-K neurons update during equilibrium phase.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("sparse_equilibrium")
class SparseEquilibrium(BioModel):
    """EqProp with sparse (Top-K) updates."""

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

        self.sparsity = 0.5
        self.criterion = nn.CrossEntropyLoss()

    def sparse_activation(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.size(1) * self.sparsity)
        top_vals, _ = torch.topk(torch.abs(x), k, dim=1)
        threshold = top_vals[:, -1].unsqueeze(1)
        mask = (torch.abs(x) >= threshold).float()
        return x * mask

    def forward(self, x: torch.Tensor, steps: int = 20, **kwargs) -> torch.Tensor:
        activations = [x]
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            activations.append(h)
        h = self.layers[-1](h)
        activations.append(h)

        for _ in range(steps):
            new_acts = [activations[0]]
            h = activations[0]

            for i, layer in enumerate(self.layers[:-1]):
                pre_activ = layer(h)
                h = self.activation(pre_activ)
                h = self.sparse_activation(h)
                new_acts.append(h)

            h = self.layers[-1](h)
            new_acts.append(h)
            activations = new_acts

        return activations[-1]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Fallback to standard Trainer loop to use persistent optimizer state (e.g., Adam/Momentum).
        Previous implementation erroneously re-initialized the optimizer every step.
        """
        return None

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
