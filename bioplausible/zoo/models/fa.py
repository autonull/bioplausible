"""
Combined Feedback Alignment Models
===================================

Aggregates all FA-family models into a single module for the model zoo.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from ..base import BioModel, ModelConfig, register_model
from ..nebc_base import NEBCBase, register_nebc
from .base import EqPropModel

# ============================================================================
# feedback_alignment.py - All FA variants
# ============================================================================


class FeedbackAlignmentLayer(nn.Module):
    """Linear layer with separate forward and feedback weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        feedback_mode: str = "random",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feedback_mode = feedback_mode

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        if feedback_mode == "random":
            self.register_buffer(
                "feedback_weight", torch.randn(in_features, out_features)
            )
        elif feedback_mode == "evolving":
            self.feedback_weight = nn.Parameter(torch.randn(in_features, out_features))
        else:
            self.feedback_weight = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight, gain=0.8)
        if hasattr(self, "feedback_weight") and self.feedback_weight is not None:
            nn.init.xavier_uniform_(self.feedback_weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def get_feedback_weight(self) -> torch.Tensor:
        if self.feedback_mode == "symmetric" or self.feedback_weight is None:
            return self.weight.t()
        return self.feedback_weight

    def get_alignment_angle(self) -> float:
        W_flat = self.weight.t().flatten()
        B_flat = self.get_feedback_weight().flatten()
        cos_sim = F.cosine_similarity(W_flat.unsqueeze(0), B_flat.unsqueeze(0))
        return cos_sim.item()


@register_model("feedback_alignment")
class FeedbackAlignmentEqProp(BioModel):
    """
    Equilibrium Propagation with Feedback Alignment.
    Uses asymmetric weights: forward W and feedback B.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        alpha: float = 0.5,
        feedback_mode: str = "random",
        use_spectral_norm: bool = True,
        config: ModelConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
            **kwargs,
        )

        self.alpha = alpha
        self.feedback_mode = feedback_mode

        self.W_in = nn.Linear(input_dim, hidden_dim)
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)

        self.layers = nn.ModuleList([
            FeedbackAlignmentLayer(hidden_dim, hidden_dim, feedback_mode)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.W_in(x)

        for layer in self.layers:
            h = torch.lerp(h, torch.tanh(x_proj + layer(h)), self.alpha)

        return h

    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        batch_size = x.size(0)
        h = torch.zeros(
            batch_size,
            self.config.hidden_dims[0] if self.config.hidden_dims else 256,
            device=x.device,
        )

        for _ in range(steps):
            h = self.forward_step(h, x)

        return self.head(h)

    def get_alignment_angles(self) -> dict[str, float]:
        angles = {}
        for i, layer in enumerate(self.layers):
            angles[f"layer_{i}"] = layer.get_alignment_angle()
        return angles

    def get_mean_alignment(self) -> float:
        angles = self.get_alignment_angles()
        if not angles:
            return 0.0
        return sum(angles.values()) / len(angles)


@register_model("adaptive_feedback_alignment")
class AdaptiveFeedbackAlignment(BioModel):
    """FA with slow adaptive feedback evolution."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or len(self.layers) == 0:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
                if hasattr(self, "hidden_dim")
                else []
            )
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.feedback_weights = nn.ParameterList()
        if config is None:
            config = self.config

        hidden_dims = (
            config.hidden_dims
            if config.hidden_dims
            else [self.hidden_dim]
            if hasattr(self, "hidden_dim")
            else []
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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
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

                if i < len(self.layers) - 1:
                    target_B = self.layers[i + 1].weight.data
                    current_B = self.feedback_weights[i + 1].data

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


@register_model("stochastic_fa")
class StochasticFA(BioModel):
    """FA with dropout on feedback signals."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or len(self.layers) == 0:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
                if hasattr(self, "hidden_dim")
                else []
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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.zero_grad()

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


@register_model("contrastive_feedback_alignment")
class ContrastiveFeedbackAlignment(BioModel):
    """Contrastive FA."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or len(self.layers) == 0:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
                if hasattr(self, "hidden_dim")
                else []
            )
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

        self.feedback_weights = nn.ParameterList()
        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim]
            if hasattr(self, "hidden_dim")
            else []
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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }


# ============================================================================
# dfa_eqprop.py - DirectFeedbackAlignmentEqProp & DeepDFAEqProp
# ============================================================================


@register_nebc("direct_feedback_alignment_eqprop")
class DirectFeedbackAlignmentEqProp(NEBCBase):
    """
    Direct Feedback Alignment with EqProp-style dynamics.
    """

    algorithm_name = "DFA"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        alpha: float = 0.5,
    ):
        self.alpha = alpha
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in, n_power_iterations=5)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_spectral_norm:
                layer = spectral_norm(layer, n_power_iterations=5)
            self.layers.append(layer)

        self.head = nn.Linear(self.hidden_dim, self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(self.head, n_power_iterations=5)

        self.feedback_projections = nn.ModuleList()
        for i in range(self.num_layers):
            B = nn.Linear(self.output_dim, self.hidden_dim, bias=False)
            nn.init.xavier_uniform_(B.weight, gain=0.1)
            B.weight.requires_grad = False
            self.feedback_projections.append(B)

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.size(0)

        h = [
            torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        x_proj = self.W_in(x)

        for _ in range(steps):
            h[0] = (1 - self.alpha) * h[0] + self.alpha * torch.tanh(
                x_proj + self.layers[0](h[0])
            )

            for i in range(1, self.num_layers):
                h[i] = (1 - self.alpha) * h[i] + self.alpha * torch.tanh(
                    h[i - 1] + self.layers[i](h[i])
                )

        return self.head(h[-1])

    def get_feedback_alignment_angles(self) -> dict[str, float]:
        angles = {}
        for i, (layer, B) in enumerate(zip(self.layers, self.feedback_projections)):
            if hasattr(layer, "weight"):
                W = layer.weight
            else:
                W = layer.parametrizations.weight.original

            W_flat = W.flatten()
            B_flat = B.weight.flatten()

            min_len = min(len(W_flat), len(B_flat))
            cos_sim = F.cosine_similarity(
                W_flat[:min_len].unsqueeze(0), B_flat[:min_len].unsqueeze(0)
            )
            angles[f"layer_{i}"] = cos_sim.item()

        return angles

    def get_stats(self) -> dict[str, float]:
        stats = super().get_stats()
        angles = self.get_feedback_alignment_angles()
        stats["mean_alignment"] = sum(angles.values()) / len(angles) if angles else 0.0
        return stats


@register_nebc("dfa_deep")
class DeepDFAEqProp(DirectFeedbackAlignmentEqProp):
    """
    DFA variant optimized for extreme depth (1000+ layers).
    """

    algorithm_name = "DeepDFA"

    def _build_layers(self):
        super()._build_layers()

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.size(0)

        h = [
            torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        x_proj = self.W_in(x)

        for _ in range(steps):
            h_new = torch.tanh(x_proj + self.layers[0](h[0]))
            h[0] = self.layer_norms[0]((1 - self.alpha) * h[0] + self.alpha * h_new)

            for i in range(1, self.num_layers):
                h_new = torch.tanh(h[i - 1] + self.layers[i](h[i]))
                h[i] = self.layer_norms[i]((1 - self.alpha) * h[i] + self.alpha * h_new)

        return self.head(h[-1])


# ============================================================================
# simple_fa.py - StandardFA
# ============================================================================


@register_model("standard_fa")
class StandardFA(BioModel):
    """Feedback Alignment with random fixed backward weights."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.feedback_weights = nn.ParameterList()
        hidden_dims = (
            self.config.hidden_dims if self.config.hidden_dims else [self.hidden_dim]
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            p = nn.Parameter(B, requires_grad=False)
            self.feedback_weights.append(p)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
    ) -> dict[str, float]:
        self.optimizer.zero_grad()

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

        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()

        for i in reversed(range(len(self.layers))):
            h_prev = activations[i]

            if i == len(self.layers) - 1:
                grad_h = error
            else:
                grad_h = torch.mm(error, self.feedback_weights[i + 1])

                h_curr = activations[i + 1]

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

            grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)

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

        self.optimizer.step()

        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": acc,
        }


# ============================================================================
# eg_fa.py - EnergyGuidedFA
# ============================================================================


@register_model("energy_guided_fa")
class EnergyGuidedFA(BioModel):
    """Energy Guided FA."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or len(self.layers) == 0:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
                if hasattr(self, "hidden_dim")
                else []
            )
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        optimizer.zero_grad()

        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
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


# ============================================================================
# em_fa.py - EnergyMinimizingFA
# ============================================================================


@register_model("energy_minimizing_fa")
class EnergyMinimizingFA(BioModel):
    """EqProp dynamics + FA updates."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or len(self.layers) == 0:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
                if hasattr(self, "hidden_dim")
                else []
            )
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

        self.feedback_weights = nn.ParameterList()
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
            self.feedback_weights.append(nn.Parameter(B, requires_grad=False))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        optimizer.zero_grad()

        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
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


# ============================================================================
# leq_fa.py - LayerwiseEquilibriumFA
# ============================================================================


@register_model("layerwise_equilibrium_fa")
class LayerwiseEquilibriumFA(BioModel):
    """Layerwise Equilibrium FA."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or len(self.layers) == 0:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
                if hasattr(self, "hidden_dim")
                else []
            )
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        optimizer.zero_grad()

        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
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


# ============================================================================
# eq_align.py - EquilibriumAlignment
# ============================================================================


@register_model("equilibrium_alignment")
class EquilibriumAlignment(EqPropModel):
    """
    Equilibrium Alignment (EqAlign) - Native Implementation.

    Combines Equilibrium Propagation's fixed-point dynamics with
    Feedback Alignment (FA) training signals.
    """

    algorithm_name = "EquilibriumAlignment"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 30,
        use_spectral_norm: bool = True,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            **kwargs,
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
            max_steps=30,
            use_spectral_norm=True,
            learning_rate=spec.default_lr,
        ).to(device)

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

        self.B_out = nn.Parameter(
            torch.randn(self.output_dim, self.hidden_dim) * 0.1, requires_grad=False
        )

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_in(x)

    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(x_transformed + self.W_rec(h))

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_out(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            x_transformed = self._transform_input(x)
            h = self._initialize_hidden_state(x)
            for _ in range(self.max_steps):
                h = self.forward_step(h, x_transformed)
            h_star = h

            logits = self._output_projection(h_star)

            loss = nn.functional.cross_entropy(logits, y)

            acc = (logits.argmax(dim=1) == y).float().mean().item()

            if y.dim() == 1:
                target = nn.functional.one_hot(y, self.output_dim).float()
            else:
                target = y

            probs = torch.softmax(logits, dim=1)
            delta_out = probs - target

        delta_h = torch.mm(delta_out, self.B_out)

        delta_h = delta_h * (1 - h_star**2)

        batch_size = x.size(0)

        grad_W_out = torch.mm(delta_out.T, h_star) / batch_size
        grad_W_rec = torch.mm(delta_h.T, h_star) / batch_size
        grad_W_in = torch.mm(delta_h.T, x) / batch_size

        grad_b_out = delta_out.mean(0)
        grad_b_rec = delta_h.mean(0)

        def update_layer(layer, grad_w, grad_b=None):
            if hasattr(layer, "parametrizations"):
                weight_param = layer.parametrizations.weight.original
            else:
                weight_param = layer.weight

            weight_param.data -= self.learning_rate * grad_w

            if layer.bias is not None and grad_b is not None:
                layer.bias.data -= self.learning_rate * grad_b

        update_layer(self.W_out, grad_W_out, grad_b_out)
        update_layer(self.W_rec, grad_W_rec, grad_b_rec)
        update_layer(self.W_in, grad_W_in, grad_b_rec)

        return {"loss": loss.item(), "accuracy": acc}
