"""
Deep Hebbian Chain with Spectral Normalization

Pure Hebbian learning in linear chains with spectral normalization.
Tests if SN enables 1000-10000 layer Hebbian networks.

Key advantage: Completely local learning (no error signals needed)
Key challenge: Norm explosion without spectral normalization

Reference: Oja, E. (1982). Simplified neuron model as a principal
component analyzer.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .nebc_base import NEBCBase, register_nebc


class HebbianLayer(nn.Module):
    """
    Single Hebbian layer with Oja's normalization rule.

    Update: ΔW = η * (y @ x.T - y^2 @ W)
    This keeps weights normalized and extracts principal components.
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

        # Weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # Use higher gain to prevent signal collapse in deep chains
        # Spectral norm will prevent explosion
        # Orthogonal initialization ensures signal preservation at depth
        nn.init.orthogonal_(self.weight, gain=1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear transformation."""
        return F.linear(x, self.weight)

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Apply Hebbian update with Oja's normalization.

        Standard Hebbian: ΔW = η * y @ x.T
        Oja's rule: ΔW = η * (y @ x.T - y^2 @ W) [keeps weights bounded]
        """
        batch_size = x.size(0)

        if hasattr(self, "weight_orig"):
            # Spectral norm is applied, update the original weights
            # We need to access the underlying parameter
            target_weight = self.weight_orig
        else:
            target_weight = self.weight

        # Optimization: In-place updates to avoid large tensor allocations
        with torch.no_grad():
            # 1. Hebbian Term: W += (lr/B) * y^T @ x
            target_weight.addmm_(y.T, x, alpha=self.learning_rate / batch_size)

            # 2. Oja Normalization: W -= lr * mean(y^2) * W
            if self.use_oja:
                # y_sq: [out, 1]
                y_sq = y.pow(2).mean(dim=0, keepdim=True).T

                # Use addcmul_ for efficient broadcasted subtraction
                # target_weight -= lr * (y_sq * self.weight)
                target_weight.addcmul_(y_sq, self.weight, value=-self.learning_rate)


@register_nebc("hebbian_chain")
class DeepHebbianChain(NEBCBase):
    """
    Deep Hebbian Chain with spectral normalization.

    Tests signal propagation through 1000+ layers with pure Hebbian learning.
    Spectral norm ensures ||W||_2 <= 1, preventing explosion.
    """

    algorithm_name = "HebbianChain"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 100,  # Default to deep
        use_spectral_norm: bool = True,
        max_steps: int = 1,  # Not iterative, just feedforward
        hebbian_lr: float = 0.001,
        use_oja: bool = True,
    ):
        self.hebbian_lr = hebbian_lr
        self.use_oja = use_oja
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

    def _build_layers(self):
        """Build deep Hebbian chain."""
        # Input projection
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in, n_power_iterations=5)

        # Deep Hebbian chain
        self.chain = nn.ModuleList()
        for i in range(self.num_layers):
            # Always use HebbianLayer
            layer = HebbianLayer(
                self.hidden_dim,
                self.hidden_dim,
                learning_rate=self.hebbian_lr,
                use_oja=self.use_oja,
            )

            if self.use_spectral_norm:
                # Apply spectral norm to the Hebbian layer
                # This renames 'weight' to 'weight_orig' and adds a hook
                layer = spectral_norm(layer, n_power_iterations=5)

            self.chain.append(layer)

        # Output projection
        self.head = nn.Linear(self.hidden_dim, self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(self.head, n_power_iterations=5)

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_signal_norms: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the chain.

        Optionally return signal norms at each layer for analysis.
        """
        if not self.training and self.use_spectral_norm:
            w = self._get_spectral_normalized_weight(self.W_in)
            b = self.W_in.bias
            h = F.linear(x, w, b)
        else:
            h = self.W_in(x)

        h = torch.tanh(h)

        norms = [h.abs().max().item()]

        for layer in self.chain:
            if not self.training and self.use_spectral_norm:
                # HebbianLayer has no bias
                w = self._get_spectral_normalized_weight(layer)
                h = F.linear(h, w)
            else:
                h = layer(h)

            h = torch.tanh(h)

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
        """
        Measure how well signals propagate through the chain.

        Returns:
            initial_norm: Signal norm after first layer
            final_norm: Signal norm after last layer
            decay_ratio: final / initial (less decay = better)
        """
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
        """Get Hebbian chain statistics."""
        stats = super().get_stats()
        stats["hebbian_lr"] = self.hebbian_lr
        stats["use_oja"] = self.use_oja
        return stats


@register_nebc("hebbian_3d")
class HebbianCube(NEBCBase):
    """
    3D Hebbian lattice for testing spatial organization.

    Inspired by Hugo de Garis "evolvable" architectures.
    Uses 3D convolutions for local connectivity in a cube.
    """

    algorithm_name = "HebbianCube"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 10,
        cube_size: int = 8,  # 8x8x8 = 512 neurons per layer
        use_spectral_norm: bool = True,
        max_steps: int = 1,
    ):
        self.cube_size = cube_size
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

    def _build_layers(self):
        """Build 3D Hebbian lattice."""
        # Project input to cube
        cube_neurons = self.cube_size**3
        self.input_proj = nn.Linear(self.input_dim, min(self.hidden_dim, cube_neurons))
        if self.use_spectral_norm:
            self.input_proj = spectral_norm(self.input_proj, n_power_iterations=5)

        # 3D convolution layers for local connectivity
        channels = max(1, self.hidden_dim // cube_neurons)
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
            if self.use_spectral_norm:
                conv = spectral_norm(conv, n_power_iterations=5)
            self.conv_layers.append(conv)

        # Flatten to output
        self.head = nn.Linear(min(self.hidden_dim, cube_neurons), self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(self.head, n_power_iterations=5)

        self._cube_neurons = min(self.hidden_dim, cube_neurons)
        self._channels = channels

    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Forward through 3D lattice."""
        batch_size = x.size(0)

        # Project to cube-compatible size
        h = self.input_proj(x)
        h = torch.tanh(h)

        # Reshape to 3D (approximately cubic)
        # If hidden_dim doesn't match cube perfectly, we adapt
        c = self._channels
        s = self.cube_size
        if h.size(1) >= c * s * s * s:
            h_3d = h[:, : c * s * s * s].view(batch_size, c, s, s, s)
        else:
            # Pad if needed
            h_padded = F.pad(h, (0, c * s * s * s - h.size(1)))
            h_3d = h_padded.view(batch_size, c, s, s, s)

        # Apply 3D convolutions
        for conv in self.conv_layers:
            h_3d = torch.tanh(conv(h_3d))

        # Flatten back
        h_flat = h_3d.view(batch_size, -1)[:, : self._cube_neurons]

        return self.head(h_flat)
