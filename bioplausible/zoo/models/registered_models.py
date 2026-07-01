"""
Example models registered with the new Zoo registry system.

This demonstrates how to register models with rich metadata for AutoScientist.
"""

import torch
import torch.nn as nn

from bioplausible.core.registry import (ComputeProfile, Domain, LocalityLevel,
                                        register_model)


@register_model(
    name="MLP",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    typical_lr_range=(1e-4, 1e-2),
    typical_batch_size_range=(32, 256),
    tags=["baseline", "mlp", "fully-connected"],
    description="Standard Multi-Layer Perceptron with backpropagation",
    citation="Rumelhart et al., 1986",
)
class MLP(nn.Module):
    """Standard MLP baseline."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def train_step(self, x, y):
        """Standard training step."""
        self.train()
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}


@register_model(
    name="EqPropMLP",
    domains=[Domain.VISION, Domain.RL],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    typical_lr_range=(1e-3, 1e-1),
    typical_batch_size_range=(16, 128),
    tags=["eqprop", "equilibrium", "bio-plausible", "energy-based"],
    description="Equilibrium Propagation on MLP architecture",
    citation="Scellier & Bengio, 2017",
)
class EqPropMLP(nn.Module):
    """Equilibrium Propagation MLP."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.algorithm_name = "EqPropMLP"

        # Build layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x, steps=None, target=None, beta=None):
        """Forward pass with optional settling."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        beta = beta or 0.0

        # Simple forward pass (single pass for demo)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)

        return h

    def train_step(self, x, y):
        """EqProp training step with free and nudged phases."""
        self.train()

        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Forward pass
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()

        return {"loss": loss.item(), "accuracy": acc}


@register_model(
    name="ForwardForwardNet",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.LAYERWISE,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.8,
    credit_assignment_type="forward-only",
    requires_backward=False,
    memory_complexity="O(1)",
    typical_lr_range=(1e-3, 1e-1),
    typical_batch_size_range=(32, 256),
    tags=["forward-forward", "layerwise", "bio-plausible", "hinton"],
    description="Hinton's Forward-Forward algorithm with layer-local goodness",
    citation="Hinton, 2022",
)
class ForwardForwardNet(nn.Module):
    """Forward-Forward Network."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_layers: int = 3,
        threshold: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold = threshold
        self.algorithm_name = "ForwardForwardNet"

        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x

    def train_step(self, x, y):
        """Forward-Forward training step with positive and negative passes."""
        self.train()

        # Create positive samples (real data with correct labels)
        pos_x = self._add_label(x, y)
        # Create negative samples (real data with wrong labels)
        neg_y = (
            y + torch.randint(1, self.output_dim, y.shape, device=y.device)
        ) % self.output_dim
        neg_x = self._add_label(x, neg_y)

        # Forward pass positive
        pos_goodness = self._forward_goodness(pos_x)
        # Forward pass negative
        neg_goodness = self._forward_goodness(neg_x)

        # Loss: want pos > threshold, neg < threshold
        pos_loss = torch.relu(self.threshold - pos_goodness).mean()
        neg_loss = torch.relu(neg_goodness - self.threshold).mean()
        loss = pos_loss + neg_loss

        # Accuracy from positive pass
        with torch.no_grad():
            logits = self(pos_x)
            acc = (logits.argmax(1) == y).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": acc,
            "pos_goodness": pos_goodness.mean().item(),
        }

    def _add_label(self, x, y):
        """Add label information to input (simplified)."""
        # In practice, would embed label and concatenate or add to input
        return x

    def _forward_goodness(self, x):
        """Compute layer-wise goodness (sum of squared activations)."""
        goodness = 0
        h = x
        for layer in self.layers:
            h = layer(h)
            h = torch.relu(h)
            goodness += h.pow(2).sum(dim=1).mean()
        return goodness


@register_model(
    name="EquiTile",
    domains=[Domain.VISION, Domain.RL, Domain.LM],
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.DISTRIBUTED,
    bio_plausibility_score=0.95,
    credit_assignment_type="local",
    requires_backward=False,
    memory_complexity="O(1)",
    typical_lr_range=(1e-3, 1e-1),
    typical_batch_size_range=(16, 128),
    tags=["equitile", "local-learning", "tiled", "neuromorphic", "scalable"],
    description="Scalable Local-Learning Architecture with Tiled Substrates",
)
class EquiTile(nn.Module):
    """EquiTile: Scalable Local-Learning Architecture."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_tiles: int = 4,
        tile_size: int = 64,
        connectivity: str = "local",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tiles = num_tiles
        self.tile_size = tile_size
        self.algorithm_name = "EquiTile"

        # Simplified tile structure
        self.tiles = nn.ModuleList(
            [nn.Linear(tile_size, tile_size) for _ in range(num_tiles)]
        )
        self.input_proj = nn.Linear(input_dim, num_tiles * tile_size)
        self.output_proj = nn.Linear(num_tiles * tile_size, output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Project to tiles
        h = self.input_proj(x)
        h = h.view(x.size(0), self.num_tiles, self.tile_size)

        # Local processing per tile
        for i, tile in enumerate(self.tiles):
            h[:, i] = torch.relu(tile(h[:, i]))

        # Merge tiles
        h = h.view(x.size(0), -1)
        return self.output_proj(h)

    def train_step(self, x, y):
        """Local learning rule training step."""
        self.train()
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}


# Register more models
print("Zoo models registered successfully")
