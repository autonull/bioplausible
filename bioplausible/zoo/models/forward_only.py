"""
Combined Forward-Only Models
=============================

Aggregates all forward-only learning models into a single module for the model zoo.
"""

import math
from typing import Dict

import torch
import torch.nn as nn

from bioplausible.core.registry import LocalityLevel
from bioplausible.core.registry import register_model

# ============================================================================
# forward_forward.py - ForwardForwardNet
# ============================================================================


class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)

    def forward(self, x):
        x_dir = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_dir, self.weight.T) + self.bias.unsqueeze(0))


@register_model(
    "forward_forward",
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.85,
    credit_assignment_type="forward-only",
    requires_backward=False,
    tags=["forward-forward", "forward-only", "local"],
    description="Forward-Forward network: trained with local goodness function.",
)
class ForwardForwardNet(nn.Module):
    """
    Hinton's Forward-Forward (2022).
    Two forward passes (positive/negative), layer-local goodness objective.
    No backward pass.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        threshold: float = 2.0,
        num_layers: int = 2,
    ):
        super().__init__()
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold = threshold

        self.layers = nn.ModuleList([FFLayer(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(FFLayer(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim * num_layers, output_dim)
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=0.01)

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers=2,
        device="cpu",
        task_type="vision",
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        ).to(device)

    def predict(self, x):
        h = x
        hidden_states = []
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)
        h_all = torch.cat(hidden_states, dim=1)
        return self.classifier(h_all)

    def forward(self, x):
        return self.predict(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        x_pos = x.clone()
        x_neg = x.clone()

        batch_size = x.shape[0]
        y_neg = torch.randint(0, self.output_dim, (batch_size,), device=x.device)
        for i in range(batch_size):
            while y_neg[i] == y[i]:
                y_neg[i] = torch.randint(0, self.output_dim, (1,)).item()

        x_pos[:, : self.output_dim] = 0.0
        x_neg[:, : self.output_dim] = 0.0
        x_pos[range(batch_size), y] = x.max()
        x_neg[range(batch_size), y_neg] = x.max()

        total_loss = 0.0
        h_pos, h_neg = x_pos, x_neg

        for layer in self.layers:
            h_pos = layer(h_pos)
            g_pos = (h_pos**2).mean(dim=1)

            h_neg = layer(h_neg)
            g_neg = (h_neg**2).mean(dim=1)

            loss = torch.log(
                1
                + torch.exp(
                    torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
                )
            ).mean()

            layer.opt.zero_grad()
            loss.backward()
            layer.opt.step()

            total_loss += loss.item()

            h_pos = h_pos.detach()
            h_neg = h_neg.detach()

        h = x
        hidden_states = []
        with torch.no_grad():
            for layer in self.layers:
                h = layer(h)
                hidden_states.append(h)
        h_all = torch.cat(hidden_states, dim=1).detach()

        logits = self.classifier(h_all)
        cls_loss = nn.functional.cross_entropy(logits, y)

        self.classifier_opt.zero_grad()
        cls_loss.backward()
        self.classifier_opt.step()

        acc = (logits.argmax(1) == y).float().mean().item()

        return {
            "loss": total_loss / len(self.layers),
            "accuracy": acc,
            "cls_loss": cls_loss.item(),
        }


# ============================================================================
# pepita.py - PEPITA
# ============================================================================


@register_model(
    "pepita",
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.8,
    credit_assignment_type="forward-only",
    requires_backward=False,
    tags=["pepita", "forward-only", "local"],
    description="PEPITA: Present the Error to Perturb the Input To modulate Activity.",
)
class PEPITA(nn.Module):
    """
    PEPITA: Present the Error to Perturb the Input To modulate Activity.
    Two forward passes; error-modulated input; no backward pass through network.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ):
        super().__init__()
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.feedback_matrix = nn.Parameter(
            torch.randn(input_dim, output_dim) / input_dim**0.5
        )
        self.lr = 0.01

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers=2,
        device="cpu",
        task_type="vision",
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        ).to(device)

    def forward(self, x, return_activations=False):
        activations = []
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
            activations.append(h)
        out = self.out_layer(h)
        if return_activations:
            return out, activations
        return out

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        y_onehot = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)

        with torch.no_grad():
            out_s, act_s = self.forward(x, return_activations=True)
            error = out_s - y_onehot

            x_mod = x + torch.mm(error, self.feedback_matrix.T)

            out_m, act_m = self.forward(x_mod, return_activations=True)

            inputs = [x] + act_s[:-1]
            for layer, a_s, a_m, inp in zip(self.layers, act_s, act_m, inputs):
                delta_a = a_m - a_s
                layer.weight.data -= self.lr * torch.mm(delta_a.T, inp) / x.shape[0]
                if layer.bias is not None:
                    layer.bias.data -= self.lr * delta_a.mean(0)

            self.out_layer.weight.data -= (
                self.lr * torch.mm(error.T, act_s[-1]) / x.shape[0]
            )
            if self.out_layer.bias is not None:
                self.out_layer.bias.data -= self.lr * error.mean(0)

        loss = (error**2).sum(1).mean().item()
        acc = (out_s.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc}
