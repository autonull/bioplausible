from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.kernel import HAS_CUPY, EqPropKernel

from ..acceleration import compile_settling_loop
from .eqprop_base import EqPropModel
from .registry import register_model
from .triton_kernel import TritonEqPropOps

# =============================================================================
# LoopedMLP - Core EqProp Model
# =============================================================================


@register_model("eqprop_mlp")
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
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        backend: str = "pytorch",  # pytorch, kernel, auto
        num_layers: int = 2, # Ignored, for compatibility
    ) -> None:
        # EqPropModel calls NEBCBase init which builds layers via _build_layers
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

        # Handle backend selection
        if backend == "auto":
            backend = "kernel" if torch.cuda.is_available() and HAS_CUPY else "pytorch"

        self.backend = backend
        self._engine = None

        if self.backend == "kernel":
            # Initialize kernel engine
            # Note: We pass use_gpu=True if CUDA is available, assuming CuPy works.
            # EqPropKernel handles fallback if CuPy import failed but HAS_CUPY checks that.
            use_gpu = HAS_CUPY and torch.cuda.is_available()
            self._engine = EqPropKernel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                max_steps=max_steps,
                use_spectral_norm=use_spectral_norm,
                use_gpu=use_gpu,
                architecture="rnn",  # Match LoopedMLP architecture
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
        """Build layers. Called by NEBCBase init."""
        # Input projection
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)

        # Recurrent (hidden-to-hidden) connection
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Output projection
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

        # Apply spectral normalization if enabled
        # CRITICAL: Only W_rec needs SN for fixed-point stability.
        # Applying it to W_in/W_out squashes signal and gradients unnecessarily.
        # Fixed: SN re-enabled after confirming torch.compile was the root cause of instability.
        if self.use_spectral_norm:
            # We keep W_in enabled for safety/reproducibility with baseline,
            # even though some literature suggests treating it as bias.
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

    def _init_weights(self) -> None:
        """Initialize weights for stable equilibrium dynamics."""
        for layer in [self.W_in, self.W_rec, self.W_out]:
            self._initialize_single_layer(layer)

    def _initialize_single_layer(self, layer: nn.Module) -> None:
        """Initialize a single layer with proper weight and bias values."""
        actual_layer = self._get_actual_layer(layer)
        if hasattr(actual_layer, "weight"):
            # Reverted to gain=0.5 for stable fixed-point dynamics required by EqProp contrastive rule.
            # gain=0.95 was too close to chaos, breaking the infinitesimal nudge assumption.
            nn.init.xavier_uniform_(actual_layer.weight, gain=0.5)
            if actual_layer.bias is not None:
                nn.init.zeros_(actual_layer.bias)

    def _get_actual_layer(self, layer: nn.Module) -> nn.Module:
        """Get the actual layer from a potentially wrapped layer."""
        if hasattr(layer, "parametrizations") and hasattr(
            layer.parametrizations, "weight"
        ):
            return layer.parametrizations.weight.original
        return layer

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor."""
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input: W_in @ x"""
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        # OPTIMIZATION: Use cached weight in eval mode
        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_in)
            b = self.W_in.bias
            return torch.nn.functional.linear(x, w, b)
        return self.W_in(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single step implementation (uncompiled)."""
        # Use Triton kernel if available for fused update
        if TritonEqPropOps.is_available():
            # pre_act = W_rec(h) + x_transformed
            # The kernel computes (1-a)h + a*tanh(pre_act)
            # Here we want straight tanh(pre_act), so alpha=1.0
            pre_act = x_transformed + self.W_rec(h)
            return TritonEqPropOps.step(h, pre_act, alpha=1.0)

        # OPTIMIZATION: Use cached weight in eval mode
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
        """Single step: h = tanh(W_in x + W_rec h)"""
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Output: W_out @ h"""
        # OPTIMIZATION: Use cached weight in eval mode
        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_out)
            b = self.W_out.bias
            return torch.nn.functional.linear(h, w, b)
        return self.W_out(h)

    def get_hebbian_pairs(self, h, x):
        """
        Return Hebbian update pairs.
        W_in connects x -> h
        W_rec connects h -> h

        Target for both is h (the equilibrium state).
        Input is x (for W_in) and h (for W_rec).
        """
        # Note: We need to use the *actual* layers, not the SpectralNorm wrappers,
        # but the forward pass uses the wrappers.
        # The generic updater calls layer(input). If layer is wrapped, it works fine.

        return [(self.W_in, x, h), (self.W_rec, h, h)]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Train step override.
        If backend is 'kernel', delegates to EqPropKernel.
        Otherwise, calls super (EqPropModel) which handles contrastive or returns None for BPTT.
        """
        if self.backend == "kernel" and self._engine is not None:
            # Convert inputs to numpy/cupy
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = y

            # Run kernel training step
            metrics = self._engine.train_step(x_np, y_np)
            return metrics

        return super().train_step(x, y)

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, Any]],
    ]:
        if self.backend == "kernel" and self._engine is not None:
            # Kernel inference
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            # Note: EqPropKernel.predict returns class indices, not logits.
            # But here forward expects logits?
            # Or we use solve_equilibrium + compute_output.

            # For compatibility with standard PyTorch workflow (e.g. cross_entropy loss external),
            # we should return logits.

            # Also need to handle steps override if possible (kernel config has max_steps)
            # The kernel stores max_steps internally.

            # Using solve_equilibrium
            h_star, _, _ = self._engine.solve_equilibrium(x_np)
            logits_np = self._engine.compute_output(h_star)

            # Convert back to tensor on same device as input
            logits = torch.from_numpy(logits_np).to(x.device)

            if return_trajectory or return_dynamics:
                # Kernel doesn't easily expose full trajectory in same format unless requested
                # Not implementing full feature parity for trajectory/dynamics in this minimal wrapper
                # unless critical.
                return logits, {} if return_dynamics else []

            return logits

        return super().forward(x, steps, return_trajectory, return_dynamics)


# =============================================================================
# BackpropMLP - Baseline for Comparison
# =============================================================================


@register_model("backprop_mlp")
class BackpropMLP(nn.Module):
    """Standard feedforward MLP for comparison (no equilibrium dynamics)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        # Fallback handling if input_dim is None (e.g. char_ngram before setup propagation)
        # But create_model generally receives valid dims from task.
        # If input_dim is explicitly None, assume it's set later or throw helpful error
        if input_dim is None:
             input_dim = 1 # Dummy fallback to prevent crash, will likely fail forward if not corrected

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        # Safe handling for num_layers <= 1
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float if needed
        if x.dtype not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.float()
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
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers
        ).to(device)
