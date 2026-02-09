"""
Standard Feedback Alignment

Random fixed backward weights for gradient approximation.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model


@register_model("dfa")
class StandardFA(BioModel):
    """Feedback Alignment with random fixed backward weights."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Random fixed feedback weights - stored as buffers
        self.feedback_weights = nn.ParameterList()
        hidden_dims = (
            self.config.hidden_dims if self.config.hidden_dims else [self.hidden_dim]
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        # Build Layers
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Feedback Matrix B (from layer i+1 to i)
            # Shape: [dims[i+1], dims[i]] (transpose of forward weight shape)
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            p = nn.Parameter(B, requires_grad=False)
            self.feedback_weights.append(p)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard forward pass."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """FA training step with random feedback."""
        self.optimizer.zero_grad()

        # Forward pass, save activations
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                activations.append(h)
            else:
                activations.append(h)

        output = activations[-1]
        loss = self.criterion(output, y)

        # Compute error at output
        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()

        # Backpropagate through RANDOM feedback weights
        for i in reversed(range(len(self.layers))):
            h_prev = activations[i]

            if i == len(self.layers) - 1:
                grad_h = error
            else:
                grad_h = torch.mm(error, self.feedback_weights[i + 1])

                # Apply derivative of activation function at layer i
                h_curr = activations[i + 1]  # layer i output (post activation)

                # Assume SiLU/ReLU/Tanh derived from activation module or config
                if isinstance(self.activation, nn.SiLU):
                    grad_h = (
                        grad_h
                        * torch.sigmoid(h_curr)
                        * (1 + h_curr * (1 - torch.sigmoid(h_curr)))
                    )
                elif isinstance(self.activation, nn.ReLU):
                    grad_h = grad_h * (h_curr > 0).float()
                elif isinstance(self.activation, nn.Tanh):
                    grad_h = grad_h * (1 - h_curr**2)
                else:
                    grad_h = grad_h * (h_curr > 0).float()

            # Weight gradient for layer i
            grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)

            # Set gradient manually for optimizer
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

            error = grad_h

        # Use optimizer step instead of manual update
        self.optimizer.step()

        # Metrics
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": acc,
        }
