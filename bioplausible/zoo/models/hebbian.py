"""
Combined Hebbian Models
=======================

Aggregates all Hebbian-family models into a single module for the model zoo.
"""

import math
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.core.registry import register_model

from ..nebc_base import NEBCBase
from ..nebc_base import register_nebc

# ============================================================================
# hebbian_chain.py - DeepHebbianChain, HebbianLayer, HebbianCube
# ============================================================================


class HebbianLayer(nn.Module):
    """
    Single Hebbian layer with Oja's normalization rule.

    Update: Delta W = eta * (y @ x.T - y^2 @ W)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.01,
        use_oja: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.use_oja = use_oja

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight, gain=1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor):
        batch_size = x.size(0)

        if hasattr(self, "weight_orig"):
            target_weight = self.weight_orig
        else:
            target_weight = self.weight

        with torch.no_grad():
            target_weight.addmm_(y.T, x, alpha=self.learning_rate / batch_size)

            if self.use_oja:
                y_sq = y.pow(2).mean(dim=0, keepdim=True).T
                target_weight.addcmul_(y_sq, self.weight, value=-self.learning_rate)


@register_model("deep_hebbian")
@register_nebc("hebbian_chain")
class DeepHebbianChain(NEBCBase):
    """
    Deep Hebbian Chain with spectral normalization.

    Tests signal propagation through 1000+ layers with pure Hebbian learning.
    """

    algorithm_name = "HebbianChain"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 100,
        use_spectral_norm: bool = True,
        max_steps: int = 1,
        hebbian_lr: float = 0.001,
        use_oja: bool = True,
    ):
        self.hebbian_lr = hebbian_lr
        self.use_oja = use_oja
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

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
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_spectral_norm=True,
            hebbian_lr=0.001,
            use_oja=True,
        ).to(device)

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in, n_power_iterations=5)

        self.chain = nn.ModuleList()
        for i in range(self.num_layers):
            layer = HebbianLayer(
                self.hidden_dim,
                self.hidden_dim,
                learning_rate=self.hebbian_lr,
                use_oja=self.use_oja,
            )

            if self.use_spectral_norm:
                layer = spectral_norm(layer, n_power_iterations=5)

            self.chain.append(layer)

        self.head = nn.Linear(self.hidden_dim, self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(self.head, n_power_iterations=5)

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_signal_norms: bool = False,
    ) -> torch.Tensor:
        if not self.training and self.use_spectral_norm:
            w = self._get_spectral_normalized_weight(self.W_in)
            b = self.W_in.bias
            h = F.linear(x, w, b)
        else:
            h = self.W_in(x)

        h.tanh_()

        norms = [h.abs().max().item()]

        for layer in self.chain:
            if not self.training and self.use_spectral_norm:
                w = self._get_spectral_normalized_weight(layer)
                h = F.linear(h, w)
            else:
                h = layer(h)

            h.tanh_()

            if return_signal_norms:
                norms.append(h.abs().max().item())

        if not self.training and self.use_spectral_norm:
            w = self._get_spectral_normalized_weight(self.head)
            b = self.head.bias
            output = F.linear(h, w, b)
        else:
            output = self.head(h)

        if return_signal_norms:
            return output, norms
        return output

    def measure_signal_propagation(self, x: torch.Tensor) -> Dict[str, float]:
        _, norms = self.forward(x, return_signal_norms=True)

        initial = norms[0]
        final = norms[-1]
        decay = final / initial if initial > 1e-10 else 0.0

        return {
            "initial_norm": initial,
            "final_norm": final,
            "decay_ratio": decay,
            "norms": norms,
        }

    def get_stats(self) -> Dict[str, float]:
        stats = super().get_stats()
        stats["hebbian_lr"] = self.hebbian_lr
        stats["use_oja"] = self.use_oja
        return stats

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        return None


@register_nebc("hebbian_3d")
class HebbianCube(NEBCBase):
    """
    3D Hebbian lattice for testing spatial organization.
    """

    algorithm_name = "HebbianCube"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 10,
        cube_size: int = 8,
        use_spectral_norm: bool = True,
        max_steps: int = 1,
    ):
        self.cube_size = cube_size
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

    def _build_layers(self):
        cube_neurons = self.cube_size**3
        self.input_proj = nn.Linear(self.input_dim, min(self.hidden_dim, cube_neurons))
        if self.use_spectral_norm:
            self.input_proj = spectral_norm(self.input_proj, n_power_iterations=5)

        channels = max(1, self.hidden_dim // cube_neurons)
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
            if self.use_spectral_norm:
                conv = spectral_norm(conv, n_power_iterations=5)
            self.conv_layers.append(conv)

        self.head = nn.Linear(min(self.hidden_dim, cube_neurons), self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(self.head, n_power_iterations=5)

        self._cube_neurons = min(self.hidden_dim, cube_neurons)
        self._channels = channels

    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        batch_size = x.size(0)

        h = self.input_proj(x)
        h = torch.tanh(h)

        c = self._channels
        s = self.cube_size
        if h.size(1) >= c * s * s * s:
            h_3d = h[:, : c * s * s * s].view(batch_size, c, s, s, s)
        else:
            h_padded = F.pad(h, (0, c * s * s * s - h.size(1)))
            h_3d = h_padded.view(batch_size, c, s, s, s)

        for conv in self.conv_layers:
            h_3d = torch.tanh(conv(h_3d))

        h_flat = h_3d.view(batch_size, -1)[:, : self._cube_neurons]

        return self.head(h_flat)


# ============================================================================
# three_factor.py - ThreeFactorHebbian
# ============================================================================


@register_model("three_factor_hebbian")
class ThreeFactorHebbian(nn.Module):
    """
    Three-Factor Learning: Delta w = eta * M * pre * post
    where M is a neuromodulatory signal (dopamine-like global reward).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ):
        super().__init__()
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.out_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        self.relu = nn.ReLU()
        self.lr = 0.005

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
        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        model = model.to(device)
        return model

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
        return self.out_layer(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        hs = [x]
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
            hs.append(h)
        out = self.out_layer(h)

        preds = out.argmax(1)
        correct = (preds == y).float()
        M = correct * 2 - 1
        M = M.to(x.device)

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                pre = hs[i]
                post = hs[i + 1]
                post_mod = post * M.unsqueeze(1)
                layer.weight.data += self.lr * torch.mm(post_mod.T, pre) / x.shape[0]

            y_onehot = torch.zeros_like(out, device=out.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            error = y_onehot - out
            self.out_layer.weight.data += (
                self.lr * torch.mm(error.T, hs[-1]) / x.shape[0]
            )

        loss = nn.functional.cross_entropy(out, y).item()
        return {"loss": loss, "accuracy": correct.mean().item()}
