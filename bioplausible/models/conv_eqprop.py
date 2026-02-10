import torch
import torch.nn as nn

from ..acceleration import compile_settling_loop
from .eqprop_base import EqPropModel
from .triton_kernel import TritonEqPropOps
from .utils import spectral_conv2d

# =============================================================================
# ConvEqProp - Convolutional EqProp for Vision Tasks
# =============================================================================


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
            input_dim=0,  # Not used directly for conv
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

        # Initialize for stability (NEBCBase calls _build_layers, but we might want extra init)
        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def _build_layers(self):
        """Build layers. Called by NEBCBase init."""
        # Input embedding
        self.embed = spectral_conv2d(
            self.input_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        # Recurrent weights
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

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, self.output_dim),
        )

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor."""
        B, _, H, W = x.shape
        return torch.zeros(
            B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input: embed(x)"""
        return self.embed(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single step implementation (uncompiled)."""
        h_norm = self.norm(h)

        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)

        h_target = ffn_out + x_transformed

        if TritonEqPropOps.is_available() and h.is_cuda:
            # Use Triton kernel for fused update: h_next = (1-gamma)*h + gamma*h_target
            return TritonEqPropOps.step_linear(h, h_target, self.gamma)
        else:
            # Use torch.lerp for more efficient interpolation
            h_next = torch.lerp(h, h_target, self.gamma)
            return h_next

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Single equilibrium iteration step.

        Args:
            h: Current hidden state
            x_transformed: Embedded input tensor (x_emb)

        Returns:
            Next hidden state
        """
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Output projection."""
        return self.head(h)
