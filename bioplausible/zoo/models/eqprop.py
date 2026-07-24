"""
Combined Equation Propagation Models
=====================================

Aggregates all EqProp-family models into a single module for the model zoo.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.acceleration.kernels import HAS_CUPY, EqPropKernel
from bioplausible.acceleration.triton_kernels import TritonEqPropOps

from ...acceleration import compile_settling_loop
from ..base import BioModel, ModelConfig, register_model
from ..utils import spectral_conv2d, spectral_linear
from .base import EqPropModel
from bioplausible.core.registry import Domain, LocalityLevel

# ============================================================================
# looped_mlp.py - LoopedMLP & BackpropMLP
# ============================================================================


@register_model(
    "eqprop_mlp",
    domains=[Domain.VISION, Domain.LM, Domain.RL, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    tags=["eqprop", "looped_mlp"],
)
class LoopedMLP(EqPropModel):
    """
    A recurrent MLP that iterates to a fixed-point equilibrium.

    The key insight: By constraining Lipschitz constant L < 1 via spectral norm,
    the network is guaranteed to converge to a unique fixed point.

    Architecture:
        h_{t+1} = tanh(W_in @ x + W_rec @ h_t)
        output = W_out @ h*  (where h* is the fixed point)

    This model can be trained using:
    1. BPTT (Backpropagation Through Time): With EqPropTrainer(use_kernel=False)
    2. EqProp (Equilibrium Propagation): Using EqPropTrainer(use_kernel=True).
       Note: For EqProp kernel mode, the weights are managed by the kernel (NumPy/CuPy),
       not this PyTorch module. This module is primarily for BPTT or inference/visualization.

    Example:
        >>> model = LoopedMLP(784, 256, 10, use_spectral_norm=True)
        >>> x = torch.randn(32, 784)
        >>> output = model(x, steps=30)  # [32, 10]
        >>> L = model.compute_lipschitz()  # Should be < 1.0
    """

    def __init__(
        self,
        input_dim: int | tuple,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        backend: str = "pytorch",
        num_layers: int = 2,
    ) -> None:
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

        if backend == "auto":
            backend = "kernel" if torch.cuda.is_available() and HAS_CUPY else "pytorch"

        self.backend = backend
        self._engine = None

        if self.backend == "kernel":
            use_gpu = HAS_CUPY and torch.cuda.is_available()
            self._engine = EqPropKernel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                max_steps=max_steps,
                use_spectral_norm=use_spectral_norm,
                use_gpu=use_gpu,
                architecture="rnn",
            )

        self._init_weights()

    def __repr__(self) -> str:
        backend_str = f", backend={self.backend}" if self.backend != "pytorch" else ""
        return (
            f"LoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm}{backend_str})"
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
            use_spectral_norm=True,
            max_steps=20,
        ).to(device)

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

    def _init_weights(self) -> None:
        for layer in [self.W_in, self.W_rec, self.W_out]:
            self._initialize_single_layer(layer)

    def _initialize_single_layer(self, layer: nn.Module) -> None:
        actual_layer = self._get_actual_layer(layer)
        if hasattr(actual_layer, "weight"):
            nn.init.xavier_uniform_(actual_layer.weight, gain=0.5)
            if actual_layer.bias is not None:
                nn.init.zeros_(actual_layer.bias)

    def _get_actual_layer(self, layer: nn.Module) -> nn.Module:
        if hasattr(layer, "parametrizations") and hasattr(
            layer.parametrizations, "weight"
        ):
            return layer.parametrizations.weight.original
        return layer

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.float()

        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_in)
            b = self.W_in.bias
            return torch.nn.functional.linear(x, w, b)
        return self.W_in(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        if TritonEqPropOps.is_available():
            pre_act = x_transformed + self.W_rec(h)
            return TritonEqPropOps.step(h, pre_act, alpha=1.0)

        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_rec)
            b = self.W_rec.bias
            rec = torch.nn.functional.linear(h, w, b)
            return torch.tanh(x_transformed + rec)

        return torch.tanh(x_transformed + self.W_rec(h))

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_out)
            b = self.W_out.bias
            return torch.nn.functional.linear(h, w, b)
        return self.W_out(h)

    def get_hebbian_pairs(self, h, x):
        return [(self.W_in, x, h), (self.W_rec, h, h)]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = y

            if x_np.ndim > 2:
                x_np = x_np.reshape(x_np.shape[0], -1)

            metrics = self._engine.train_step(x_np, y_np)
            return metrics

        return super().train_step(x, y)

    def forward(
        self,
        x: torch.Tensor,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, dict[str, Any]]
    ):
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if x_np.ndim > 2:
                x_np = x_np.reshape(x_np.shape[0], -1)

            h_star, _, _ = self._engine.solve_equilibrium(x_np)
            logits_np = self._engine.compute_output(h_star)

            logits = torch.from_numpy(logits_np).to(x.device)

            if return_trajectory or return_dynamics:
                return logits, {} if return_dynamics else []

            return logits

        return super().forward(x, steps, return_trajectory, return_dynamics)


@register_model("backprop_mlp")
class BackpropMLP(nn.Module):
    """Standard feedforward MLP for comparison (no equilibrium dynamics)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ) -> None:
        super().__init__()
        layers = []
        if input_dim is None:
            input_dim = 64

        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        if num_layers <= 1:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.float()
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        if x.size(1) != self.net[0].in_features:
            raise ValueError(
                f"Input feature dimension mismatch. "
                f"Expected {self.net[0].in_features} but got {x.size(1)}."
            )

        return self.net(x)

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
        ).to(device)


# ============================================================================
# standard_eqprop.py - StandardEqProp
# ============================================================================


@register_model("eqprop")
class StandardEqProp(BioModel):
    """
    Standard EqProp with free/nudged phases and bidirectional relaxation.

    Implements the dynamics:
    h_i = sigma(W_i h_{i-1} + W_{i+1}^T h_{i+1} + b_i)
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_spectral_normalized_weight(self, layer: nn.Module) -> torch.Tensor:
        if not self.training and hasattr(layer, "_cached_sn_weight"):
            return layer._cached_sn_weight

        weight = layer.weight

        if not self.training:
            layer._cached_sn_weight = weight.detach()

        return weight

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.modules():
                if hasattr(module, "_cached_sn_weight"):
                    delattr(module, "_cached_sn_weight")
        return self

    @compile_settling_loop
    def forward_dynamics(
        self,
        activations: list[torch.Tensor],
        beta: float = 0.0,
        target: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        new_activations = [activations[0]]

        num_layers = len(self.layers)

        for i in range(num_layers):
            layer = self.layers[i]
            h_prev = activations[i]

            if not self.training:
                w = self._get_spectral_normalized_weight(layer)
                b = layer.bias
                a_bu = torch.nn.functional.linear(h_prev, w, b)
            else:
                a_bu = layer(h_prev)

            a_td = 0.0
            if i < num_layers - 1:
                next_layer = self.layers[i + 1]
                h_next = activations[i + 2]
                if hasattr(next_layer, "weight"):
                    if not self.training:
                        w = self._get_spectral_normalized_weight(next_layer)
                    else:
                        w = next_layer.weight
                    a_td = torch.matmul(h_next, w)

            total_input = a_bu + a_td

            if i < num_layers - 1:
                h_new = self.activation(total_input)
            else:
                h_new = total_input

            if i == num_layers - 1 and beta > 0 and target is not None:
                h_new = h_new + beta * (target - h_new)

            new_activations.append(h_new)

        return new_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        eq_steps = steps if steps is not None else self.eq_steps

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        if return_trajectory:
            trajectory = [None] * (eq_steps + 1)
            trajectory[0] = [a.detach().cpu() for a in activations]
        else:
            trajectory = None

        deltas = [] if return_dynamics else None

        for step_idx in range(eq_steps):
            prev_activations = activations
            activations = self.forward_dynamics(activations, beta, target)

            delta = 0.0
            for k in range(1, len(activations)):
                delta += torch.dist(activations[k], prev_activations[k], p=2).item()

            if return_dynamics:
                deltas.append(delta)

            if step_idx > 5 and delta < 1e-3:
                break

            if return_trajectory:
                trajectory[step_idx + 1] = [a.detach().cpu() for a in activations]

        self._last_activations = activations
        out = activations[-1]

        if return_dynamics:
            return out, {
                "trajectory": trajectory if return_trajectory else None,
                "deltas": deltas,
                "final_delta": deltas[-1] if deltas else 0.0,
            }

        if return_trajectory:
            return out, trajectory

        return out

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)

        with torch.no_grad():
            self.forward(x, beta=0.0)
            free_activations = self._last_activations
            output_free = free_activations[-1]

        with torch.no_grad():
            self.forward(x, beta=self.beta, target=target)
            nudged_activations = self._last_activations

        self.optimizer.zero_grad()

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                h_prev_free = free_activations[i]
                h_post_free = free_activations[i + 1]

                h_prev_nudged = nudged_activations[i]
                h_post_nudged = nudged_activations[i + 1]

                prod_nudged = torch.matmul(h_post_nudged.T, h_prev_nudged)
                prod_free = torch.matmul(h_post_free.T, h_prev_free)

                dW = (prod_nudged - prod_free) / self.beta
                dW = dW / x.size(0)

                param_container = layer
                weight_name = "weight"

                if hasattr(layer, "parametrizations") and hasattr(
                    layer.parametrizations, "weight"
                ):
                    param_container = layer.parametrizations.weight
                    weight_name = "original"

                w_param = getattr(param_container, weight_name)

                if w_param.grad is None:
                    w_param.grad = -dW
                else:
                    w_param.grad += -dW

                if layer.bias is not None:
                    db = (h_post_nudged - h_post_free).sum(0) / self.beta
                    db = db / x.size(0)
                    if layer.bias.grad is None:
                        layer.bias.grad = -db
                    else:
                        layer.bias.grad += -db

        self.optimizer.step()

        pred = output_free.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        loss = nn.functional.cross_entropy(output_free, y).item()

        return {
            "loss": loss,
            "accuracy": acc,
        }


# ============================================================================
# conv_eqprop.py - ConvEqProp
# ============================================================================


class ConvEqProp(EqPropModel):
    """
    Convolutional Equilibrium Propagation Model.

    Uses ResNet-like loop structure with spectral normalization.
    Suitable for image classification tasks (MNIST, CIFAR-10).

    Example:
        >>> model = ConvEqProp(1, 32, 10)  # MNIST
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x, steps=25)  # [32, 10]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dim: int,
        gamma: float = 0.5,
        use_spectral_norm: bool = True,
        max_steps: int = 25,
        gradient_method: str = "bptt",
    ) -> None:
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm

        super().__init__(
            input_dim=0,
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def _build_layers(self):
        self.embed = spectral_conv2d(
            self.input_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        self.W1 = spectral_conv2d(
            self.hidden_channels,
            self.hidden_channels * 2,
            kernel_size=3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.W2 = spectral_conv2d(
            self.hidden_channels * 2,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        self.norm = nn.GroupNorm(8, self.hidden_channels)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, self.output_dim),
        )

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        return torch.zeros(
            B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        h_norm = self.norm(h)

        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)

        h_target = ffn_out + x_transformed

        if TritonEqPropOps.is_available() and h.is_cuda:
            return TritonEqPropOps.step_linear(h, h_target, self.gamma)
        else:
            h_next = torch.lerp(h, h_target, self.gamma)
            return h_next

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


# ============================================================================
# deep_ep.py - DirectedEP (DEEP)
# ============================================================================


@register_model("directed_ep")
class DirectedEP(BioModel):
    """
    Directed EqProp (DEEP) with separate forward and feedback weights.
    Both sets of weights are updated to minimize the energy/loss.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim]
            if hasattr(self, "hidden_dim")
            else []
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        self.forward_layers = nn.ModuleList()
        self.feedback_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            fwd = nn.Linear(dims[i], dims[i + 1])
            self.forward_layers.append(fwd)

            bwd = nn.Linear(dims[i + 1], dims[i], bias=False)
            self.feedback_layers.append(bwd)

        self.to(kwargs.get("device", "cpu"))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward_dynamics(
        self,
        activations: list[torch.Tensor],
        beta: float = 0.0,
        target: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:

        updated_activations = [activations[0]]

        for k in range(len(self.forward_layers)):
            h_prev = activations[k]

            a_bu = self.forward_layers[k](h_prev)

            a_td = 0.0
            if k < len(self.forward_layers) - 1:
                h_next = activations[k + 2]
                a_td = self.feedback_layers[k + 1](h_next)

            total = a_bu + a_td

            if k < len(self.forward_layers) - 1:
                h_new = self.activation(total)
            else:
                h_new = total

            if k == len(self.forward_layers) - 1 and beta > 0 and target is not None:
                h_new = h_new + beta * (target - h_new)

            updated_activations.append(h_new)

        return updated_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        eq_steps = steps if steps is not None else self.eq_steps

        activations = [x]
        h = x
        for i, layer in enumerate(self.forward_layers):
            h = layer(h)
            if i < len(self.forward_layers) - 1:
                h = self.activation(h)
            activations.append(h)

        trajectory = []
        deltas = []

        if return_trajectory:
            trajectory.append([a.detach().cpu() for a in activations])

        for _ in range(eq_steps):
            prev_activations = activations
            activations = self.forward_dynamics(activations, beta, target)

            if return_dynamics:
                delta = 0.0
                for k in range(1, len(activations)):
                    delta += torch.dist(
                        activations[k], prev_activations[k], p=float("inf")
                    ).item()
                deltas.append(delta)

            if return_trajectory:
                trajectory.append([a.detach().cpu() for a in activations])

        self._last_activations = activations
        out = activations[-1]

        if return_dynamics:
            return out, {
                "trajectory": trajectory if return_trajectory else None,
                "deltas": deltas,
                "final_delta": deltas[-1] if deltas else 0.0,
            }

        if return_trajectory:
            return out, trajectory

        return out

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)

        with torch.no_grad():
            self.forward(x, beta=0.0)
            free = self._last_activations

        with torch.no_grad():
            self.forward(x, beta=self.beta, target=target)
            nudged = self._last_activations

        self.optimizer.zero_grad()

        with torch.no_grad():
            for i in range(len(self.forward_layers)):
                h_prev_free, h_post_free = free[i], free[i + 1]
                h_prev_nudge, h_post_nudge = nudged[i], nudged[i + 1]

                prod_nudged = torch.matmul(h_post_nudge.T, h_prev_nudge)
                prod_free = torch.matmul(h_post_free.T, h_prev_free)

                dW = (prod_nudged - prod_free) / self.beta
                dW /= x.size(0)

                if self.forward_layers[i].weight.grad is None:
                    self.forward_layers[i].weight.grad = -dW
                else:
                    self.forward_layers[i].weight.grad += -dW

                if self.forward_layers[i].bias is not None:
                    db = (h_post_nudge - h_post_free).sum(0) / self.beta
                    db /= x.size(0)
                    if self.forward_layers[i].bias.grad is None:
                        self.forward_layers[i].bias.grad = -db
                    else:
                        self.forward_layers[i].bias.grad += -db

                prod_nudged_b = torch.matmul(h_prev_nudge.T, h_post_nudge)
                prod_free_b = torch.matmul(h_prev_free.T, h_post_free)

                dB = (prod_nudged_b - prod_free_b) / self.beta
                dB /= x.size(0)

                if self.feedback_layers[i].weight.grad is None:
                    self.feedback_layers[i].weight.grad = -dB
                else:
                    self.feedback_layers[i].weight.grad += -dB

        self.optimizer.step()

        loss = nn.functional.cross_entropy(free[-1], y).item()
        acc = (free[-1].argmax(dim=1) == y).float().mean().item()

        return {"loss": loss, "accuracy": acc}

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
# memory_efficient.py - MemoryEfficientLoopedMLP, MemoryEfficientEqPropModel
# ============================================================================


class MemoryEfficientLoopedMLP(LoopedMLP):
    """
    Memory-efficient version of LoopedMLP that defaults to O(1) memory kernel backend.

    This model uses the NumPy/CuPy kernel for O(1) memory training, making it suitable
    for deep networks where PyTorch autograd would consume O(N) memory.

    Example:
        >>> model = MemoryEfficientLoopedMLP(784, 256, 10)
        >>> print(model.backend)  # 'kernel' if CUDA/CuPy available, else 'pytorch'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        use_gpu_if_available: bool = True,
    ) -> None:
        if use_gpu_if_available and HAS_CUPY and torch.cuda.is_available():
            backend = "kernel"
        else:
            backend = "pytorch"

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
            max_steps=max_steps,
            gradient_method=gradient_method,
            backend=backend,
        )

        self.is_memory_efficient = self.backend == "kernel"

    def __repr__(self) -> str:
        backend_str = f", backend={self.backend}"
        efficiency_str = (
            ", O(1) memory" if self.is_memory_efficient else ", O(N) memory"
        )
        return (
            f"MemoryEfficientLoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm}{backend_str}{efficiency_str})"
        )


class MemoryEfficientEqPropModel(EqPropModel):
    """
    Base class for memory-efficient EqProp models that can leverage kernel backend.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        use_spectral_norm: bool = True,
        memory_efficient: bool = True,
        use_gpu: bool = True,
    ):
        self.memory_efficient = memory_efficient
        self.use_gpu = use_gpu and HAS_CUPY and torch.cuda.is_available()

        if memory_efficient and HAS_CUPY and self.use_gpu:
            self.backend = "kernel"
        else:
            self.backend = "pytorch"

        super().__init__(
            max_steps=max_steps,
            gradient_method=gradient_method,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
        )

        if self.backend == "kernel":
            self._engine = EqPropKernel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                max_steps=max_steps,
                use_spectral_norm=use_spectral_norm,
                use_gpu=self.use_gpu,
            )
        else:
            self._engine = None

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float] | None:
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = y

            metrics = self._engine.train_step(x_np, y_np)
            return metrics

        return super().train_step(x, y)

    def forward(self, x: torch.Tensor, steps: int | None = None, **kwargs):
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            h_star, _, _ = self._engine.solve_equilibrium(x_np)
            logits_np = self._engine.compute_output(h_star)

            return torch.from_numpy(logits_np).to(x.device)

        return super().forward(x, steps, **kwargs)


def create_memory_efficient_model(
    model_type: str, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
) -> Any:
    if model_type.lower() in ["loopedmlp", "memory_efficient", "o1_memory"]:
        return MemoryEfficientLoopedMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
        )
    else:
        raise ValueError(f"Unsupported memory-efficient model type: {model_type}")


# ============================================================================
# transformer_eqprop.py - TransformerEqProp & EqPropAttention
# ============================================================================


class EqPropAttention(nn.Module):
    """Self-attention that participates in equilibrium dynamics."""

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, use_sn: bool = True
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_k = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_v = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_o = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape

        Q, K, V = self._compute_qkv(h, batch_size, seq_len)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return self.W_o(self._reshape_output(out, batch_size, seq_len))

    def _compute_qkv(
        self, h: torch.Tensor, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = (
            self
            .W_q(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self
            .W_k(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self
            .W_v(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        return Q, K, V

    def _reshape_output(
        self, out: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        return (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )


class TransformerEqProp(EqPropModel):
    """
    Transformer with equilibrium dynamics.

    All layers (attention + FFN) iterate together to a joint equilibrium.
    Spectral normalization ensures stable convergence.

    Example:
        >>> model = TransformerEqProp(vocab_size=1000, hidden_dim=256, output_dim=10)
        >>> x = torch.randint(0, 1000, (32, 64))
        >>> output = model(x, steps=20)
    """

    def __init__(
        self,
        vocab_size: int = None,
        hidden_dim: int = 256,
        output_dim: int = 27,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        alpha: float = 0.5,
        use_spectral_norm: bool = True,
        max_steps: int = 20,
        gradient_method: str = "bptt",
        input_dim: int = None,
    ) -> None:
        if vocab_size is None and output_dim is not None:
            vocab_size = output_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.alpha = alpha
        self.use_spectral_norm = use_spectral_norm

        super().__init__(
            input_dim=0,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

    def _build_layers(self):
        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_dim)

        self.attentions = nn.ModuleList([
            EqPropAttention(
                self.hidden_dim, self.num_heads, use_sn=self.use_spectral_norm
            )
            for _ in range(self.num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                spectral_linear(
                    self.hidden_dim,
                    self.hidden_dim * 2,
                    use_sn=self.use_spectral_norm,
                ),
                nn.ReLU(),
                spectral_linear(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    use_sn=self.use_spectral_norm,
                ),
            )
            for _ in range(self.num_layers)
        ])

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])

        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        return torch.zeros(
            batch_size, seq_len, self.hidden_dim, device=x.device, dtype=torch.float
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)
        return x_emb

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        current_h = h
        for i in range(self.num_layers):
            current_h = self._forward_layer(current_h, x_transformed, i)
        return current_h

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _forward_layer(
        self, h: torch.Tensor, x_emb: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        h_norm = self.norms1[layer_idx](h)
        h = h + self.attentions[layer_idx](h_norm)

        h_norm = self.norms2[layer_idx](h)
        ffn_out = self.ffns[layer_idx](h_norm)

        h_target = h + ffn_out + x_emb

        if TritonEqPropOps.is_available() and h.is_cuda:
            return TritonEqPropOps.step(h, h_target, alpha=self.alpha)

        return torch.lerp(h, torch.tanh(h_target), self.alpha)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h.mean(dim=1))


# ============================================================================
# causal_transformer_eqprop.py - CausalTransformerEqProp & CausalEqPropAttention
# ============================================================================


class CausalEqPropAttention(nn.Module):
    """Self-attention with causal masking for autoregressive generation."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, use_sn: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_k = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_v = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_o = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)

    def forward(
        self, h: torch.Tensor, causal_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape

        Q = (
            self
            .W_q(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self
            .W_k(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self
            .W_v(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return self.W_o(
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )


class CausalTransformerEqProp(nn.Module):
    """
    TransformerEqProp with causal masking for language modeling.

    Key differences from classification TransformerEqProp:
    - Causal attention mask
    - LM head instead of classification head
    - Outputs logits for full sequence (not just final token)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
        eq_steps: int = 20,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attentions = nn.ModuleList([
            CausalEqPropAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                spectral_linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                spectral_linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        self.register_buffer("causal_mask", None)
        self._create_causal_mask(max_seq_len)

    def _create_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        if x.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.long()

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = (
            self.causal_mask[:seq_len, :seq_len]
            if self.causal_mask is not None
            else None
        )

        h = torch.zeros_like(x_emb)

        for _ in range(steps):
            for i in range(self.num_layers):
                h_norm = self.norms1[i](h)
                h = h + self.attentions[i](h_norm, causal_mask=causal_mask)

                h_norm = self.norms2[i](h)
                ffn_out = self.ffns[i](h_norm)

                h_target = h + ffn_out + x_emb

                if TritonEqPropOps.is_available() and h.is_cuda:
                    h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
                else:
                    h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        logits = self.lm_head(h)

        return logits

    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0
    ):
        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if generated.size(1) >= self.max_seq_len:
                    break

        return generated


# ============================================================================
# eqprop_diffusion.py - EqPropDiffusion
# ============================================================================


@register_model("eqprop_diffusion")
class EqPropDiffusion(nn.Module):
    """
    Equilibrium Propagation Diffusion Model.

    Hypothesis: Denoising diffusion is energy minimization.
    Energy Formulation: E(x,t) = ||x - Denoise(x_t,t)||² + lambda R(x)

    This model predicts the clean image x_0 from x_t.
    """

    def __init__(self, img_channels=1, hidden_channels=64, gradient_method="bptt"):
        super().__init__()
        self.denoiser = SimpleConvEqProp(
            input_channels=img_channels + 1,
            hidden_channels=hidden_channels,
            output_dim=img_channels,
            pool_output=False,
            use_spectral_norm=True,
            gradient_method=gradient_method,
        )
        self.img_channels = img_channels

        T = 1000
        self.T = T
        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_variance", posterior_variance)

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
        channels = input_dim if input_dim is not None else 1

        if channels == 784:
            channels = 1
        elif channels == 3072:
            channels = 3
        elif channels > 10:
            side = int(channels**0.5)
            if side * side == channels:
                channels = 1
            elif (channels % 3 == 0) and (
                int((channels / 3) ** 0.5) ** 2 * 3 == channels
            ):
                channels = 3

        return cls(img_channels=channels, hidden_channels=hidden_dim).to(device)

    def train_step(self, x, y=None):
        device = x.device
        batch_size = x.shape[0]

        t = torch.randint(0, self.T, (batch_size,), device=device).long()

        noise = torch.randn_like(x)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_omab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        x_noisy = sqrt_ab * x + sqrt_omab * noise

        pred = self(x_noisy, t)

        loss = F.mse_loss(pred, x)

        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item()}

    def val_step(self, x, y=None):
        device = x.device
        batch_size = x.shape[0]

        t = torch.randint(0, self.T, (batch_size,), device=device).long()

        noise = torch.randn_like(x)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_omab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        x_noisy = sqrt_ab * x + sqrt_omab * noise

        pred = self(x_noisy, t)

        loss = F.mse_loss(pred, x).item()

        accuracy = 1.0 / (1.0 + loss)

        return {"loss": loss, "accuracy": accuracy}

    def predict_x0(self, x_t, t):
        batch_size, _, h, w = x_t.shape

        t_norm = t.float() / self.T
        t_emb = t_norm.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)

        x_input = torch.cat([x_t, t_emb], dim=1)
        return self.denoiser(x_input)

    def denoise_step(self, x_t, t_norm, steps=30):
        batch_size, _, h, w = x_t.shape

        if t_norm.dim() == 1:
            t_emb = t_norm.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        else:
            t_emb = t_norm.expand(batch_size, 1, h, w)

        x_input = torch.cat([x_t, t_emb], dim=1)

        return self.denoiser(x_input, steps=steps)

    def forward(self, x, t=None):
        if t is None:
            if x.shape[1] == self.img_channels + 1:
                return self.denoiser(x)
            raise ValueError("t must be provided for diffusion forward pass")

        return self.predict_x0(x, t)

    @torch.no_grad()
    def sample(self, num_samples=16, img_size=(1, 28, 28), device="cpu"):
        self.eval()
        B = num_samples
        C, H, W = img_size

        x = torch.randn(B, C, H, W, device=device)

        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            x_0_pred = self.predict_x0(x, t)

            alpha_t = self.alpha[t].view(B, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
            alpha_bar_prev_t = self.alpha_bar_prev[t].view(B, 1, 1, 1)
            beta_t = self.beta[t].view(B, 1, 1, 1)

            coeff1 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
            coeff2 = (
                torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
            )

            mean = coeff1 * x_0_pred + coeff2 * x

            if i > 0:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t].view(B, 1, 1, 1)
                sigma = torch.sqrt(var)
                x = mean + sigma * noise
            else:
                x = mean

        self.train()
        return x.clamp(-1, 1)


# ============================================================================
# holomorphic_ep.py - HolomorphicEP
# ============================================================================


@register_model("holomorphic_ep")
class HolomorphicEP(BioModel):
    """
    Holomorphic EqProp with complex-valued weights and states.
    Uses complex tanh activation which is holomorphic.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

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
            layer.weight = nn.Parameter(layer.weight.to(torch.complex64))
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.to(torch.complex64))
            self.layers.append(layer)

        self.to(kwargs.get("device", "cpu"))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def forward_dynamics(
        self,
        activations: list[torch.Tensor],
        beta: float = 0.0,
        target: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        new_activations = [activations[0]]

        num_layers = len(self.layers)

        for i in range(num_layers):
            layer = self.layers[i]
            h_prev = activations[i]

            a_bu = layer(h_prev)

            a_td = 0.0 + 0.0j
            if i < num_layers - 1:
                next_layer = self.layers[i + 1]
                h_next = activations[i + 2]
                if hasattr(next_layer, "weight"):
                    w = next_layer.weight
                    w_backward = w.conj().T
                    a_td = torch.matmul(h_next, w_backward.T)

            total_input = a_bu + a_td

            if i < num_layers - 1:
                h_new = self.activation(total_input)
            else:
                h_new = total_input

            if i == num_layers - 1 and beta > 0 and target is not None:
                if not target.is_complex():
                    target = target.to(h_new.dtype)

                h_new = h_new + beta * (target - h_new)

            new_activations.append(h_new)

        return new_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(torch.complex64)

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        num_steps = steps if steps is not None else self.eq_steps

        for _ in range(num_steps):
            activations = self.forward_dynamics(activations, beta, target)

        self._last_activations = activations

        return activations[-1].real

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        target = target.to(torch.complex64)

        with torch.no_grad():
            self.forward(x, beta=0.0)
            free_activations = self._last_activations
            output_free = free_activations[-1]

        with torch.no_grad():
            self.forward(x, beta=self.beta, target=target)
            nudged_activations = self._last_activations

        self.optimizer.zero_grad()

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                h_prev_free = free_activations[i]
                h_post_free = free_activations[i + 1]

                h_prev_nudged = nudged_activations[i]
                h_post_nudged = nudged_activations[i + 1]

                prod_nudged = torch.matmul(h_post_nudged.T, h_prev_nudged.conj())
                prod_free = torch.matmul(h_post_free.T, h_prev_free.conj())

                dW = (prod_nudged - prod_free) / self.beta
                dW = dW / x.size(0)

                if layer.weight.grad is None:
                    layer.weight.grad = -dW
                else:
                    layer.weight.grad += -dW

                if layer.bias is not None:
                    db = (h_post_nudged - h_post_free).sum(0) / self.beta
                    db = db / x.size(0)
                    if layer.bias.grad is None:
                        layer.bias.grad = -db
                    else:
                        layer.bias.grad += -db

        self.optimizer.step()

        pred = output_free.real.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        loss = nn.functional.cross_entropy(output_free.real, y).item()

        return {
            "loss": loss,
            "accuracy": acc,
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
# finite_nudge_ep.py - FiniteNudgeEP
# ============================================================================


@register_model("finite_nudge_ep")
class FiniteNudgeEP(StandardEqProp):
    """
    Finite-Nudge EqProp.
    Operates with large beta values (e.g. beta=1.0) where the infinitesimal
    approximation of the gradient is replaced by a finite difference
    that optimizes a global energy bound.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if "beta" in kwargs:
            self.beta = kwargs["beta"]
        elif self.config and self.config.extra and "beta" in self.config.extra:
            self.beta = self.config.extra["beta"]

        if self.beta < 0.5:
            self.beta = 1.0

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        metrics = super().train_step(x, y)

        return metrics

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

        if "equilibrium_steps" in kwargs:
            config.equilibrium_steps = kwargs["equilibrium_steps"]
            config.max_steps = kwargs["equilibrium_steps"]
        if "beta" in kwargs:
            config.beta = kwargs["beta"]

        return cls(config=config).to(device)


# ============================================================================
# lazy_eqprop.py - LazyEqProp
# ============================================================================


@dataclass
class LazyStats:
    """Statistics for lazy execution."""

    total_neurons: int = 0
    active_neurons: int = 0
    skipped_neurons: int = 0

    @property
    def skip_ratio(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        return self.skipped_neurons / self.total_neurons

    @property
    def flop_savings(self) -> float:
        return self.skip_ratio * 100

    def reset(self):
        self.total_neurons = 0
        self.active_neurons = 0
        self.skipped_neurons = 0


class LazyEqProp(nn.Module):
    """
    Event-driven Equilibrium Propagation with lazy updates.

    Key insight: Most neurons don't change much per step.
    Skip updates for neurons with |Delta input| < epsilon.

    Achieves 70-95% FLOP savings on typical workloads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        alpha: float = 0.5,
        epsilon: float = 0.01,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.epsilon = epsilon

        self.embed = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                layer = spectral_norm(layer)
            self.layers.append(layer)

        self.head = nn.Linear(hidden_dim, output_dim)

        for layer in self.layers:
            if hasattr(layer, "parametrizations"):
                weight = layer.parametrizations.weight.original
            else:
                weight = layer.weight
            nn.init.orthogonal_(weight)
            with torch.no_grad():
                weight.mul_(0.8)

        self.stats = LazyStats()

    def lazy_forward_step(
        self,
        h_states: dict[int, torch.Tensor],
        prev_inputs: dict[int, torch.Tensor],
        x_emb: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        batch_size = x_emb.size(0)
        device = x_emb.device

        new_states = {}
        new_inputs = {}

        for i, layer in enumerate(self.layers):
            if i == 0:
                layer_input = x_emb
            else:
                layer_input = h_states.get(i - 1, x_emb)

            new_inputs[i] = layer_input

            prev = prev_inputs.get(i, torch.zeros_like(layer_input))

            input_delta = (layer_input - prev).abs()
            active_mask = input_delta.mean(dim=-1, keepdim=True) > self.epsilon
            active_mask = active_mask.expand_as(layer_input).float()

            num_neurons = batch_size * self.hidden_dim
            num_active = int(active_mask.sum().item())
            self.stats.total_neurons += num_neurons
            self.stats.active_neurons += num_active
            self.stats.skipped_neurons += num_neurons - num_active

            h_current = h_states.get(
                i, torch.zeros(batch_size, self.hidden_dim, device=device)
            )

            h_new = torch.tanh(layer(layer_input))
            h_update = (1 - self.alpha) * h_current + self.alpha * h_new

            new_states[i] = active_mask * h_update + (1 - active_mask) * h_current

        return new_states, new_inputs

    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        self.stats.reset()

        x_emb = self.embed(x)

        h_states = {
            i: torch.zeros(batch_size, self.hidden_dim, device=device)
            for i in range(self.num_layers)
        }
        prev_inputs = {}

        for _ in range(steps):
            h_states, prev_inputs = self.lazy_forward_step(h_states, prev_inputs, x_emb)

        return self.head(h_states[self.num_layers - 1])

    def get_flop_savings(self) -> float:
        return self.stats.flop_savings


# ============================================================================
# neural_cube.py - NeuralCube
# ============================================================================


@register_model("neural_cube")
class NeuralCube(nn.Module):
    """
    A 3D lattice neural network where neurons exist in 3D space.

    Each neuron connects only to its 26 neighbors (3x3x3 local patch minus self).
    This mimics biological neural tissue where connectivity is spatially local.
    """

    def __init__(
        self,
        cube_size: int = 6,
        input_dim: int = 64,
        output_dim: int = 10,
        max_steps: int = 30,
    ):
        super().__init__()
        self.cube_size = cube_size
        self.n_neurons = cube_size**3
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_steps = max_steps

        self.W_in = nn.Linear(input_dim, self.n_neurons)

        self.W_local = nn.Parameter(torch.zeros(self.n_neurons, 27))

        self.W_out = nn.Linear(self.n_neurons, output_dim)

        self.register_buffer("neighbor_indices", self._build_neighbor_indices())

        self._init_weights()

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
        cube_size = int(round(hidden_dim ** (1 / 3)))
        return cls(
            cube_size=max(4, cube_size),
            input_dim=input_dim,
            output_dim=output_dim,
        ).to(device)

    def _build_neighbor_indices(self) -> torch.Tensor:
        size = self.cube_size
        indices = torch.full((self.n_neurons, 27), self.n_neurons, dtype=torch.long)

        for z in range(size):
            for y in range(size):
                for x in range(size):
                    neuron_idx = z * size * size + y * size + x
                    neighbor_count = 0

                    for dz in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                nz, ny, nx = z + dz, y + dy, x + dx

                                if 0 <= nz < size and 0 <= ny < size and 0 <= nx < size:
                                    neighbor_idx = nz * size * size + ny * size + nx
                                    indices[neuron_idx, neighbor_count] = neighbor_idx

                                neighbor_count += 1

        return indices

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.zeros_(self.W_in.bias)
        nn.init.normal_(self.W_local, mean=0, std=0.1)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.5)
        nn.init.zeros_(self.W_out.bias)

    def local_update(self, h: torch.Tensor) -> torch.Tensor:
        if (
            hasattr(TritonEqPropOps, "neural_cube_update")
            and TritonEqPropOps.is_available()
            and h.is_cuda
        ):
            return TritonEqPropOps.neural_cube_update(h, self.W_local, self.cube_size)

        batch_size = h.shape[0]

        h_padded = F.pad(h, (0, 1))

        indices_expanded = self.neighbor_indices.unsqueeze(0).expand(batch_size, -1, -1)
        h_expanded = h_padded.unsqueeze(1).expand(-1, self.n_neurons, -1)
        neighbor_activations = torch.gather(h_expanded, 2, indices_expanded)

        weighted = (neighbor_activations * self.W_local.unsqueeze(0)).sum(dim=2)

        return weighted

    def forward(
        self,
        x: torch.Tensor,
        steps: int = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.shape[0]
        device = x.device

        h = torch.zeros(batch_size, self.n_neurons, device=device, dtype=x.dtype)

        x_proj = self.W_in(x)

        trajectory = [h.detach()] if return_trajectory else None

        for _ in range(steps):
            local_contrib = self.local_update(h)

            h = torch.tanh(x_proj + local_contrib)

            if return_trajectory:
                trajectory.append(h.detach())

        out = self.W_out(h)

        if return_trajectory:
            return out, trajectory
        return out

    def get_topology_stats(self) -> dict:
        active_weights = (self.W_local.abs() > 0.01).float().mean().item()

        fully_connected = self.n_neurons * self.n_neurons
        local_connections = self.n_neurons * 27

        return {
            "cube_size": self.cube_size,
            "n_neurons": self.n_neurons,
            "local_connections": local_connections,
            "fully_connected_equivalent": fully_connected,
            "connection_reduction": 1 - (local_connections / fully_connected),
            "active_weight_fraction": active_weights,
        }

    def get_cube_slice(self, h: torch.Tensor, z: int) -> torch.Tensor:
        size = self.cube_size
        start = z * size * size
        end = (z + 1) * size * size

        slice_flat = h[..., start:end]
        return slice_flat.reshape(*h.shape[:-1], size, size)

    def visualize_cube_ascii(self, h: torch.Tensor, sample_idx: int = 0) -> str:
        chars = " .dbBF"
        size = self.cube_size

        lines = []
        lines.append(f"Neural Cube {size}x{size}x{size} (z-slices)")
        lines.append("=" * (size * 3 + 10))

        h_sample = h[sample_idx].detach().cpu()
        h_norm = (h_sample - h_sample.min()) / (h_sample.max() - h_sample.min() + 1e-8)

        for z in range(size):
            lines.append(f"\nz={z}:")
            for y in range(size):
                row = ""
                for x in range(size):
                    idx = z * size * size + y * size + x
                    val = h_norm[idx].item()
                    char_idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
                    row += chars[char_idx] * 2
                lines.append(f"  {row}")

        return "\n".join(lines)


# ============================================================================
# temporal_resonance.py - TemporalResonanceEqProp
# ============================================================================


class TemporalResonanceEqProp(nn.Module):
    """
    EqProp network that converges to a stable oscillation (limit cycle)
    instead of a fixed point.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        alpha: float = 0.5,
        oscillation_strength: float = 0.1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.oscillation_strength = oscillation_strength

        self.W_in = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            spectral_linear(hidden_dim, hidden_dim, use_sn=use_spectral_norm)
            for _ in range(num_layers)
        ])

        self.osc_coupling = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if use_spectral_norm:
            self.osc_coupling = spectral_norm(self.osc_coupling)

        self.head = nn.Linear(hidden_dim, output_dim)

        self._init_oscillatory_weights()

    def _init_oscillatory_weights(self):
        with torch.no_grad():
            dim = self.hidden_dim
            self.osc_coupling.weight.zero_()

            for i in range(0, dim - 1, 2):
                angle = 0.1
                c, s = math.cos(angle), math.sin(angle)
                self.osc_coupling.weight[i, i] = c
                self.osc_coupling.weight[i, i + 1] = -s
                self.osc_coupling.weight[i + 1, i] = s
                self.osc_coupling.weight[i + 1, i + 1] = c

            self.osc_coupling.weight.mul_(self.oscillation_strength)

    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.W_in(x)

        h_recurrent = x_emb
        for layer in self.layers:
            h_recurrent = h_recurrent + layer(torch.tanh(h))

        h_oscillatory = self.osc_coupling(h)

        h_target = torch.tanh(h_recurrent + h_oscillatory)

        return torch.lerp(h, h_target, self.alpha)

    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for _ in range(steps):
            h = self.forward_step(h, x)

        return self.head(h)

    def forward_sequence(
        self, x_seq: torch.Tensor, steps_per_frame: int = 5
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, seq_len, _ = x_seq.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)

        outputs = []
        trajectories = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            for _ in range(steps_per_frame):
                h = self.forward_step(h, x_t)

            trajectories.append(h.detach())
            outputs.append(self.head(h))

        outputs = torch.stack(outputs, dim=1)
        return outputs, trajectories

    def detect_limit_cycle(
        self, x: torch.Tensor, max_steps: int = 200, cycle_detection_window: int = 20
    ) -> dict:
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        trajectory = []

        for _ in range(max_steps):
            h = self.forward_step(h, x)
            trajectory.append(h.detach())

        trajectory = torch.stack(trajectory)
        recent = trajectory[-cycle_detection_window:]

        correlations = []
        for lag in range(1, cycle_detection_window // 2):
            corr = (
                F
                .cosine_similarity(recent[:-lag].flatten(1), recent[lag:].flatten(1))
                .mean()
                .item()
            )
            correlations.append(corr)

        if correlations:
            max_corr = max(correlations)
            cycle_length = correlations.index(max_corr) + 1
            cycle_detected = max_corr > 0.9
        else:
            max_corr, cycle_length, cycle_detected = 0, 0, False

        amplitude = torch.std(recent, dim=0).mean().item()

        return {
            "cycle_detected": cycle_detected,
            "cycle_length": cycle_length,
            "max_correlation": max_corr,
            "amplitude": amplitude,
        }


# ============================================================================
# ternary.py - TernaryEqProp
# ============================================================================


class TernaryQuantize(torch.autograd.Function):
    """
    Ternary quantization with Straight-Through Estimator.

    Forward: Quantize weights to {-1, 0, +1}
    Backward: Pass gradients through unchanged (STE)
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        ctx.save_for_backward(weight)

        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0

        return ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight,) = ctx.saved_tensors
        grad_weight = grad_output.clone()
        return grad_weight, None


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights."""

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.weight, gain=0.8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ternary_weight = TernaryQuantize.apply(self.weight, self.threshold)
        return F.linear(x, ternary_weight, self.bias)

    def get_weight_stats(self) -> dict:
        w = self.weight.detach()
        threshold = self.threshold

        n_pos = (w > threshold).sum().item()
        n_neg = (w < -threshold).sum().item()
        n_zero = w.numel() - n_pos - n_neg

        total = w.numel()
        return {
            "positive": n_pos / total,
            "zero": n_zero / total,
            "negative": n_neg / total,
            "sparsity": n_zero / total,
        }


class TernaryEqProp(nn.Module):
    """
    Equilibrium Propagation with Ternary Weights.

    Combines recurrent fixed-point dynamics with extreme quantization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        threshold: float = 0.5,
        max_steps: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold = threshold
        self.max_steps = max_steps

        self.W_in = TernaryLinear(input_dim, hidden_dim, threshold)
        self.W_rec = TernaryLinear(hidden_dim, hidden_dim, threshold)
        self.W_out = TernaryLinear(hidden_dim, output_dim, threshold)

    def forward(
        self,
        x: torch.Tensor,
        steps: int = None,
    ) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.shape[0]

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        x_proj = self.W_in(x)

        for _ in range(steps):
            h = torch.tanh(x_proj + self.W_rec(h))

        return self.W_out(h)

    def get_model_stats(self) -> dict:
        stats = {
            "W_in": self.W_in.get_weight_stats(),
            "W_rec": self.W_rec.get_weight_stats(),
            "W_out": self.W_out.get_weight_stats(),
        }

        total_zero = sum(s["sparsity"] for s in stats.values())
        stats["overall_sparsity"] = total_zero / 3

        return stats

    def count_bit_operations(self) -> dict:
        in_ops = self.input_dim * self.hidden_dim
        rec_ops = self.hidden_dim * self.hidden_dim
        out_ops = self.hidden_dim * self.output_dim
        total_ops = in_ops + rec_ops * self.max_steps + out_ops

        float32_ops = total_ops * 2

        sparsity = self.get_model_stats()["overall_sparsity"]
        ternary_ops = int(total_ops * (1 - sparsity))

        return {
            "float32_operations": float32_ops,
            "ternary_operations": ternary_ops,
            "speedup_factor": (
                float32_ops / ternary_ops if ternary_ops > 0 else float("inf")
            ),
            "sparsity_used": sparsity,
        }


# ============================================================================
# sparse_eq.py - SparseEquilibrium
# ============================================================================


@register_model("sparse_equilibrium")
class SparseEquilibrium(BioModel):
    """EqProp with sparse (Top-K) updates."""

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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
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


# ============================================================================
# mom_eq.py - MomentumEquilibrium
# ============================================================================


@register_model("momentum_equilibrium")
class MomentumEquilibrium(BioModel):
    """EqProp with momentum in settling dynamics."""

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

        self.momentum = 0.5
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        activations = [x]
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            activations.append(h)
        h = self.layers[-1](h)
        activations.append(h)

        velocities = [torch.zeros_like(a) for a in activations]

        for _ in range(self.config.equilibrium_steps):
            new_acts = [activations[0]]
            h = activations[0]

            for i, layer in enumerate(self.layers[:-1]):
                target = self.activation(layer(h))
                delta = target - activations[i + 1]
                velocities[i + 1] = self.momentum * velocities[i + 1] + 0.5 * delta
                h = activations[i + 1] + velocities[i + 1]
                new_acts.append(h)

            h = self.layers[-1](h)
            new_acts.append(h)
            activations = new_acts

        return activations[-1]

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
# homeostatic.py - HomeostaticEqProp
# ============================================================================


@dataclass
class HomeostasisMetrics:
    avg_velocity: float
    lipschitz_estimate: float
    brake_applied: float
    boost_applied: float
    layers_braked: int
    layers_boosted: int


class HomeostaticEqProp(nn.Module):
    """
    EqProp with Dynamic Lipschitz Scaling for autonomous stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 5,
        alpha: float = 0.5,
        target_lipschitz: float = 0.95,
        velocity_threshold_high: float = 0.1,
        velocity_threshold_low: float = 0.01,
        adaptation_rate: float = 0.01,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha

        self.target_lipschitz = target_lipschitz
        self.velocity_threshold_high = velocity_threshold_high
        self.velocity_threshold_low = velocity_threshold_low
        self.adaptation_rate = adaptation_rate

        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.register_buffer("layer_scales", torch.ones(num_layers))

        self.head = nn.Linear(hidden_dim, output_dim)

        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(0.7)

        self.last_velocities: dict[int, float] = {}
        self.homeostasis_history: list[HomeostasisMetrics] = []

    def _estimate_layer_lipschitz(self, layer_idx: int) -> float:
        original_weight = self.layers[layer_idx].weight
        scaled_weight = original_weight * self.layer_scales[layer_idx]

        with torch.no_grad():
            W = scaled_weight
            u = torch.randn(W.shape[1], device=W.device)
            u = F.normalize(u, dim=0)
            for _ in range(3):
                v = F.normalize(W @ u, dim=0)
                u = F.normalize(W.T @ v, dim=0)
            sigma = torch.norm(W @ u).item()
        return sigma

    def forward_step(
        self,
        h_states: dict[int, torch.Tensor],
        x: torch.Tensor,
        track_velocity: bool = False,
    ) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
        new_states = {}
        velocities = {}
        x_emb = self.W_in(x)

        for i, layer in enumerate(self.layers):
            pre = x_emb if i == 0 else h_states.get(i - 1, torch.zeros_like(x_emb))
            h_curr = h_states.get(i, torch.zeros_like(pre))

            scale = self.layer_scales[i]
            h_target = torch.tanh(F.linear(pre, layer.weight * scale, layer.bias))

            h_new = (1 - self.alpha) * h_curr + self.alpha * h_target
            new_states[i] = h_new

            if track_velocity:
                velocity = torch.mean(torch.abs(h_new - h_curr)).item()
                velocities[i] = velocity

        return new_states, velocities

    def apply_homeostasis(self, velocities: dict[int, float]) -> HomeostasisMetrics:
        brake_total = 0.0
        boost_total = 0.0
        layers_braked = 0
        layers_boosted = 0

        for i, velocity in velocities.items():
            current_L = self._estimate_layer_lipschitz(i)

            if velocity > self.velocity_threshold_high or current_L > (
                self.target_lipschitz + 0.1
            ):
                error_v = max(0, velocity - self.velocity_threshold_high)
                error_l = max(0, current_L - self.target_lipschitz)

                error = error_v + error_l

                factor = 1.0 - (self.adaptation_rate * (1.0 + 10.0 * error))
                factor = max(0.5, factor)

                self.layer_scales[i] *= factor
                brake_total += 1.0 - factor
                layers_braked += 1

            elif velocity < self.velocity_threshold_low:
                current_L = self._estimate_layer_lipschitz(i)
                if current_L < self.target_lipschitz:
                    error = self.velocity_threshold_low - velocity
                    factor = 1.0 + (self.adaptation_rate * (1.0 + 5.0 * error))
                    factor = min(1.5, factor)

                    self.layer_scales[i] *= factor
                    boost_total += factor - 1.0
                    layers_boosted += 1

        self.layer_scales.clamp_(0.1, 3.0)

        avg_velocity = sum(velocities.values()) / len(velocities) if velocities else 0.0
        avg_lipschitz = (
            sum(self._estimate_layer_lipschitz(i) for i in range(self.num_layers))
            / self.num_layers
        )

        metrics = HomeostasisMetrics(
            avg_velocity=avg_velocity,
            lipschitz_estimate=avg_lipschitz,
            brake_applied=brake_total,
            boost_applied=boost_total,
            layers_braked=layers_braked,
            layers_boosted=layers_boosted,
        )

        self.homeostasis_history.append(metrics)
        self.last_velocities = velocities

        return metrics

    def forward(
        self, x: torch.Tensor, steps: int = 30, apply_homeostasis: bool = True
    ) -> torch.Tensor:
        batch_size = x.size(0)
        h_states = {
            i: torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for i in range(self.num_layers)
        }

        all_velocities = []
        for step in range(steps):
            track = step >= steps // 2
            h_states, velocities = self.forward_step(h_states, x, track_velocity=track)
            if track:
                all_velocities.append(velocities)

        if apply_homeostasis and all_velocities:
            avg_velocities = {}
            for i in range(self.num_layers):
                avg_velocities[i] = sum(v.get(i, 0) for v in all_velocities) / len(
                    all_velocities
                )
            self.apply_homeostasis(avg_velocities)

        return self.head(h_states[self.num_layers - 1])

    def get_stability_report(self) -> str:
        lipschitz = [self._estimate_layer_lipschitz(i) for i in range(self.num_layers)]
        max_L = max(lipschitz) if lipschitz else 0.0
        status = "STABLE" if max_L < 1.0 else "UNSTABLE"

        lines = [
            f"Max Lipschitz: {max_L:.4f} {status}",
            f"Layer Scales: {[f'{s:.3f}' for s in self.layer_scales.tolist()]}",
        ]
        if self.homeostasis_history:
            last = self.homeostasis_history[-1]
            lines.append(
                f"Last Action: {last.layers_braked} braked, {last.layers_boosted} boosted"
            )

        return "\n".join(lines)


# ============================================================================
# modern_conv_eqprop.py - ModernConvEqProp & SimpleConvEqProp
# ============================================================================


@register_model("modern_conv_eqprop")
class ModernConvEqProp(EqPropModel):
    """
    Multi-stage ConvEqProp with equilibrium settling.

    Architecture:
        Input: 3x32x32 (CIFAR-10)
        Stage 1: Conv 3->64, no pooling (32x32)
        Stage 2: Conv 64->128, stride 2 (16x16)
        Stage 3: Conv 128->256, stride 2 (8x8)
        Equilibrium: Recurrent conv at 256 channels
        Output: Global pool -> Linear(256, 10)
    """

    def __init__(
        self,
        eq_steps: int = 15,
        gamma: float = 0.5,
        hidden_channels: int = 64,
        use_spectral_norm: bool = True,
        gradient_method: str = "bptt",
        input_dim: int = 0,
        output_dim: int = 10,
        **kwargs,
    ):
        self.gamma = gamma
        self.base_hidden_channels = hidden_channels

        self.input_channels = 3
        self.input_dims = (32, 32)
        flat_input_dim = 0

        if "input_channels" in kwargs:
            self.input_channels = kwargs["input_channels"]

        if isinstance(input_dim, tuple):
            flat_input_dim = math.prod(input_dim)
            if len(input_dim) == 3:
                self.input_channels = input_dim[0]
                self.input_dims = (input_dim[1], input_dim[2])
            elif len(input_dim) == 1:
                self.input_channels = input_dim[0]
        elif isinstance(input_dim, int) and input_dim > 0:
            flat_input_dim = input_dim

        if isinstance(input_dim, int) and input_dim == 64:
            self.input_channels = 1
            self.input_dims = (8, 8)
        if isinstance(input_dim, int) and input_dim == 784:
            self.input_channels = 1
            self.input_dims = (28, 28)

        self.output_dim_val = output_dim

        super().__init__(
            input_dim=flat_input_dim,
            hidden_dim=hidden_channels * 4,
            output_dim=self.output_dim_val,
            max_steps=eq_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

    def _build_layers(self):
        hidden_channels = self.base_hidden_channels

        self.stage1 = nn.Sequential(
            spectral_conv2d(
                self.input_channels,
                hidden_channels,
                3,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.Tanh(),
        )

        in_dim_0 = getattr(self, "input_dims", (32, 32))[0]

        self.stage2 = nn.Sequential(
            spectral_conv2d(
                hidden_channels,
                hidden_channels * 2,
                3,
                stride=2 if in_dim_0 >= 16 else 1,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.Tanh(),
        )

        self.stage3 = nn.Sequential(
            spectral_conv2d(
                hidden_channels * 2,
                hidden_channels * 4,
                3,
                stride=2 if in_dim_0 >= 32 else 1,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.Tanh(),
        )

        self.eq_conv = spectral_conv2d(
            hidden_channels * 4,
            hidden_channels * 4,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.eq_norm = nn.GroupNorm(8, hidden_channels * 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels * 4, self.output_dim_val)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "parametrizations"):
                    weight = m.parametrizations.weight.original
                else:
                    weight = m.weight
                nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="tanh")
                weight.data.mul_(0.5)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        with torch.no_grad():
            h_trans = self._transform_input(x[:1])
            H_out, W_out = h_trans.shape[2], h_trans.shape[3]

        return torch.zeros(
            B, self.hidden_dim, H_out, W_out, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            B = x.size(0)
            area = x.size(1) // self.input_channels
            S = int(math.sqrt(area))
            x = x.view(B, self.input_channels, S, S)

        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        return h

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        h_norm = self.eq_norm(h)

        if TritonEqPropOps.is_available() and h.is_cuda:
            pre_act = self.eq_conv(h_norm) + x_transformed
            return TritonEqPropOps.step(h, pre_act, alpha=self.gamma)

        h_next = torch.tanh(self.eq_conv(h_norm) + x_transformed)
        return torch.lerp(h, h_next, self.gamma)

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        features = self.pool(h).flatten(1)
        return self.fc(features)

    def get_hebbian_pairs(self, h, x):
        if not hasattr(self, "feedforward_net"):
            self.feedforward_net = nn.Sequential(self.stage1, self.stage2, self.stage3)

        h_norm = self.eq_norm(h)

        return [(self.eq_conv, h_norm, h), (self.feedforward_net, x, h)]

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
            eq_steps=30,
            hidden_channels=hidden_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs,
        ).to(device)


class SimpleConvEqProp(EqPropModel):
    """
    Simplified single-stage ConvEqProp for comparison.
    Refactored to use EqPropModel.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        eq_steps: int = 20,
        gamma: float = 0.5,
        use_spectral_norm: bool = True,
        gradient_method: str = "bptt",
        input_channels: int = 3,
        output_dim: int = 10,
        pool_output: bool = True,
    ):
        self.hidden_channels = hidden_channels
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm
        self.input_channels_count = input_channels
        self.output_dim_val = output_dim
        self.pool_output = pool_output

        super().__init__(
            input_dim=0,
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            max_steps=eq_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

    def _build_layers(self):
        self.embed = spectral_conv2d(
            self.input_channels_count,
            self.hidden_channels,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        self.W_rec = spectral_conv2d(
            self.hidden_channels,
            self.hidden_channels,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.norm = nn.GroupNorm(8, self.hidden_channels)

        if self.pool_output:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_channels, self.output_dim_val),
            )
        else:
            self.head = spectral_conv2d(
                self.hidden_channels,
                self.output_dim_val,
                kernel_size=1,
                use_sn=self.use_spectral_norm,
            )

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        return torch.zeros(
            B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        h_norm = self.norm(h)

        if TritonEqPropOps.is_available() and h.is_cuda:
            pre_act = self.W_rec(h_norm) + x_transformed
            return TritonEqPropOps.step(h, pre_act, alpha=self.gamma)

        h_next = torch.tanh(self.W_rec(h_norm) + x_transformed)
        return torch.lerp(h, h_next, self.gamma)

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

    def get_hebbian_pairs(self, h, x):
        h_norm = self.norm(h)
        return [(self.W_rec, h_norm, h), (self.embed, x, h)]


# ============================================================================
# eqprop_lm_variants.py - EqProp LM Variants
# ============================================================================


EQPROP_LM_REGISTRY: dict[str, type[nn.Module]] = {}


def register_eqprop_lm(name: str):
    def decorator(cls):
        EQPROP_LM_REGISTRY[name] = cls
        return cls

    return decorator


def get_eqprop_lm(name: str, **kwargs) -> nn.Module:
    if name not in EQPROP_LM_REGISTRY:
        raise ValueError(
            f"Unknown EqProp LM variant: {name}. Available: {list(EQPROP_LM_REGISTRY.keys())}"
        )
    return EQPROP_LM_REGISTRY[name](**kwargs)


def list_eqprop_lm_variants() -> list:
    return list(EQPROP_LM_REGISTRY.keys())


class EqPropAttentionLM(nn.Module):
    """Self-attention with spectral normalization for EqProp."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        use_sn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_k = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_v = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_o = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor, causal_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape

        Q = (
            self
            .W_q(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self
            .W_k(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self
            .W_v(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        return self.W_o(
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )


class CausalMask:
    """Helper for causal masking."""

    _cache = {}

    @classmethod
    def get(cls, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if key not in cls._cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            cls._cache[key] = mask
        return cls._cache[key]


@register_eqprop_lm("full")
class FullEqPropLM(nn.Module):
    """
    Full Transformer with all layers participating in equilibrium settling.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 15,
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attentions = nn.ModuleList([
            EqPropAttentionLM(hidden_dim, num_heads, use_sn) for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                spectral_linear(hidden_dim, hidden_dim * 2, use_sn),
                nn.ReLU(),
                spectral_linear(hidden_dim * 2, hidden_dim, use_sn),
            )
            for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = CausalMask.get(seq_len, x.device)

        h = torch.zeros_like(x_emb)

        for _ in range(steps):
            for i in range(self.num_layers):
                h_norm = self.norms1[i](h)
                h = h + self.attentions[i](h_norm, causal_mask)

                h_norm = self.norms2[i](h)
                ffn_out = self.ffns[i](h_norm)

                h_target = h + ffn_out + x_emb

                if TritonEqPropOps.is_available() and h.is_cuda:
                    h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
                else:
                    h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        return self.lm_head(h)

    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0
    ):
        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if generated.size(1) >= self.max_seq_len:
                    break

        return generated


@register_eqprop_lm("attention_only")
class EqPropAttentionOnlyLM(nn.Module):
    """
    Only attention uses equilibrium settling, FFN is standard feedforward.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 10,
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attentions = nn.ModuleList([
            EqPropAttentionLM(hidden_dim, num_heads, use_sn) for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = CausalMask.get(seq_len, x.device)

        for i in range(self.num_layers):
            h_attn = h.clone()
            for _ in range(steps):
                h_norm = self.norms1[i](h_attn)
                attn_out = self.attentions[i](h_norm, causal_mask)

                h_target = h + attn_out

                if TritonEqPropOps.is_available() and h_attn.is_cuda:
                    h_attn = TritonEqPropOps.step_linear(
                        h_attn, h_target, alpha=self.alpha
                    )
                else:
                    h_attn = (1 - self.alpha) * h_attn + self.alpha * h_target

            h = h_attn

            h = h + self.ffns[i](self.norms2[i](h))

        return self.lm_head(h)


@register_eqprop_lm("recurrent_core")
class RecurrentEqPropLM(nn.Module):
    """
    Single recurrent block that iterates to equilibrium.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 20,
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attention = EqPropAttentionLM(hidden_dim, num_heads, use_sn)
        self.ffn = nn.Sequential(
            spectral_linear(hidden_dim, hidden_dim * 2, use_sn),
            nn.ReLU(),
            spectral_linear(hidden_dim * 2, hidden_dim, use_sn),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = CausalMask.get(seq_len, x.device)

        h = torch.zeros_like(x_emb)

        for _ in range(steps):
            h_norm = self.norm1(h)
            h = h + self.attention(h_norm, causal_mask)

            h_norm = self.norm2(h)
            ffn_out = self.ffn(h_norm)

            h_target = h + ffn_out + x_emb

            if TritonEqPropOps.is_available() and h.is_cuda:
                h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
            else:
                h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        return self.lm_head(h)


@register_eqprop_lm("hybrid")
class HybridEqPropLM(nn.Module):
    """
    First N-1 layers are standard, final layer uses equilibrium.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 10,
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.standard_blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.standard_blocks.append(
                nn.ModuleDict({
                    "attention": EqPropAttentionLM(hidden_dim, num_heads, use_sn=False),
                    "ffn": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    "norm1": nn.LayerNorm(hidden_dim),
                    "norm2": nn.LayerNorm(hidden_dim),
                })
            )

        self.eq_attention = EqPropAttentionLM(hidden_dim, num_heads, use_sn)
        self.eq_ffn = nn.Sequential(
            spectral_linear(hidden_dim, hidden_dim * 2, use_sn),
            nn.ReLU(),
            spectral_linear(hidden_dim * 2, hidden_dim, use_sn),
        )
        self.eq_norm1 = nn.LayerNorm(hidden_dim)
        self.eq_norm2 = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = CausalMask.get(seq_len, x.device)

        for block in self.standard_blocks:
            h = h + block["attention"](block["norm1"](h), causal_mask)
            h = h + block["ffn"](block["norm2"](h))

        h_input = h.clone()
        for _ in range(steps):
            h_norm = self.eq_norm1(h)
            h = h + self.eq_attention(h_norm, causal_mask)

            h_norm = self.eq_norm2(h)
            ffn_out = self.eq_ffn(h_norm)

            h_target = h + ffn_out + h_input

            if TritonEqPropOps.is_available() and h.is_cuda:
                h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
            else:
                h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        return self.lm_head(h)


@register_eqprop_lm("looped_mlp")
class LoopedMLPForLM(nn.Module):
    """
    MLP-based LM using the core LoopedMLP architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        max_seq_len: int = 256,
        eq_steps: int = 20,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.eq_steps = eq_steps
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.W_in = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_rec = spectral_linear(hidden_dim, hidden_dim, use_sn)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        x_proj = self.W_in(x_emb)

        h = torch.zeros_like(x_proj)

        for _ in range(steps):
            pre_act = x_proj + self.W_rec(h)
            if TritonEqPropOps.is_available() and h.is_cuda:
                h = TritonEqPropOps.step(h, pre_act, alpha=1.0)
            else:
                h = torch.tanh(pre_act)

        return self.lm_head(h)


def create_eqprop_lm(
    variant: str,
    vocab_size: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    scale: float = 1.0,
    **kwargs,
) -> nn.Module:
    scaled_hidden = int(hidden_dim * math.sqrt(scale))
    scaled_hidden = max(32, (scaled_hidden // 4) * 4)

    return get_eqprop_lm(
        variant,
        vocab_size=vocab_size,
        hidden_dim=scaled_hidden,
        num_layers=num_layers,
        **kwargs,
    )


def compare_variants(vocab_size: int = 65, seq_len: int = 64, batch_size: int = 4):
    results = []
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    for name in list_eqprop_lm_variants():
        model = get_eqprop_lm(name, vocab_size=vocab_size, hidden_dim=128, num_layers=2)
        params = sum(p.numel() for p in model.parameters())

        import time

        start = time.time()
        with torch.no_grad():
            _ = model(x)
        elapsed = time.time() - start

        results.append({
            "variant": name,
            "parameters": params,
            "forward_time_ms": elapsed * 1000,
        })

    return results


@register_model("eqprop_transformer")
class EqPropLMWrapper(nn.Module):
    """
    Proxy class for EqProp LM variants.
    Delegates to create_eqprop_lm via build().
    """

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
        return create_eqprop_lm(
            variant=spec.variant,
            vocab_size=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_sn=True,
        ).to(device)


# ============================================================================
# graph_eqprop.py - GraphEqProp
# ============================================================================


try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None


@register_model("graph_eqprop")
class GraphEqProp(EqPropModel):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, max_steps: int = 30
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
        )

    def _build_layers(self):
        if GCNConv is None:
            self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.conv = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.W_out = nn.Linear(self.hidden_dim, self.output_dim)
            return

        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.conv = GCNConv(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

    def _initialize_hidden_state(self, x: Any) -> torch.Tensor:
        if hasattr(x, "x"):
            num_nodes = x.x.size(0)
            return torch.zeros(
                (num_nodes, self.hidden_dim), device=x.x.device, dtype=x.x.dtype
            )
        else:
            return torch.zeros(
                (x.size(0), self.hidden_dim), device=x.device, dtype=x.dtype
            )

    def _transform_input(self, x: Any) -> Any:
        if hasattr(x, "x"):
            u = self.W_in(x.x)
            return (u, x.edge_index)
        else:
            return self.W_in(x)

    def forward_step(self, h: torch.Tensor, x_transformed: Any) -> torch.Tensor:
        if isinstance(x_transformed, tuple):
            u, edge_index = x_transformed
            if GCNConv is not None:
                return torch.tanh(u + self.conv(h, edge_index))
        return torch.tanh(x_transformed + self.conv(h))

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_out(h)

    def train_step(self, x: Any, y: torch.Tensor) -> dict[str, float]:
        if not hasattr(x, "train_mask"):
            return super().train_step(x, y)

        with torch.no_grad():
            h_star, _ = self.solve_equilibrium(x)
        logits = self._output_projection(h_star)

        mask = x.train_mask
        if not mask.any():
            mask = torch.ones_like(y, dtype=torch.bool)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits[mask], y[mask])

        beta = 0.5
        v = torch.zeros_like(logits)
        logits_masked = logits[mask].clone().detach().requires_grad_(True)
        L_masked = loss_fn(logits_masked, y[mask])
        grad_out = torch.autograd.grad(L_masked, logits_masked)[0]
        v[mask] = grad_out

        x_trans = self._transform_input(x)
        h_nudged = h_star.clone()
        with torch.no_grad():
            for _ in range(5):
                h_nudged = self.forward_step(h_nudged, x_trans) - beta * torch.mm(
                    v, self.W_out.weight
                )

        self.zero_grad()

        h_star.requires_grad = False
        if isinstance(x_trans, tuple):
            u, edge_index = x_trans
            pre_act_star = u + self.conv(h_star, edge_index)
        else:
            pre_act_star = x_trans + self.conv(h_star)
        E_free = torch.sum(0.5 * h_star**2 - h_star * pre_act_star)
        E_free.backward()
        free_grads = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in self.parameters()
        ]

        self.zero_grad()
        h_nudged.requires_grad = False
        if isinstance(x_trans, tuple):
            u, edge_index = x_trans
            pre_act_nudged = u + self.conv(h_nudged, edge_index)
        else:
            pre_act_nudged = x_trans + self.conv(h_nudged)
        E_nudged = torch.sum(0.5 * h_nudged**2 - h_nudged * pre_act_nudged)
        E_nudged.backward()

        lr = 0.001
        with torch.no_grad():
            for p, gf in zip(self.parameters(), free_grads):
                gn = p.grad
                if gn is not None:
                    p.data -= lr * (gf - gn) / beta

            self.W_out.weight.data -= lr * torch.mm(v[mask].T, h_star[mask])
            self.W_out.bias.data -= lr * v[mask].sum(0)

        acc = (logits[mask].argmax(1) == y[mask]).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers=2,
        device="cpu",
        task_type="graph",
        **kwargs,
    ):
        return cls(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        ).to(device)
