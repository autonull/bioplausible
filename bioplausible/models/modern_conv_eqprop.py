"""
Modern Convolutional EqProp for CIFAR-10 (Track 34)

Multi-stage convolutional architecture with equilibrium settling.
Target: 75%+ accuracy on CIFAR-10 (vs 44.5% baseline with LoopedMLP).

Architecture inspired by ResNet with spectral normalization for stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from ..acceleration import compile_settling_loop
from .eqprop_base import EqPropModel
from .triton_kernel import TritonEqPropOps
from .utils import spectral_conv2d


class ModernConvEqProp(EqPropModel):
    """
    Multi-stage ConvEqProp with equilibrium settling.

    Architecture:
        Input: 3×32×32 (CIFAR-10)
        Stage 1: Conv 3→64, no pooling (32×32)
        Stage 2: Conv 64→128, stride 2 (16×16)
        Stage 3: Conv 128→256, stride 2 (8×8)
        Equilibrium: Recurrent conv at 256 channels
        Output: Global pool → Linear(256, 10)

    Key Features:
    - All convolutions use spectral normalization
    - GroupNorm instead of BatchNorm (better for small batches)
    - Equilibrium settling only in deepest stage (efficient)
    """

    def __init__(
        self,
        eq_steps: int = 15,
        gamma: float = 0.5,
        hidden_channels: int = 64,
        use_spectral_norm: bool = True,
        gradient_method: str = "bptt",
    ):
        self.gamma = gamma
        self.base_hidden_channels = hidden_channels

        super().__init__(
            input_dim=0,  # Not used directly
            hidden_dim=hidden_channels * 4,  # Deepest layer dim
            output_dim=10,
            max_steps=eq_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

    def _build_layers(self):
        """Build layers. Called by NEBCBase init."""
        hidden_channels = self.base_hidden_channels

        # Stage 1: Initial feature extraction (32×32)
        self.stage1 = nn.Sequential(
            spectral_conv2d(
                3, hidden_channels, 3, padding=1, use_sn=self.use_spectral_norm
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.Tanh(),
        )

        # Stage 2: Downsample to 16×16
        self.stage2 = nn.Sequential(
            spectral_conv2d(
                hidden_channels,
                hidden_channels * 2,
                3,
                stride=2,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.Tanh(),
        )

        # Stage 3: Downsample to 8×8
        self.stage3 = nn.Sequential(
            spectral_conv2d(
                hidden_channels * 2,
                hidden_channels * 4,
                3,
                stride=2,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.Tanh(),
        )

        # Equilibrium recurrent block (operates at 8×8 spatial resolution)
        self.eq_conv = spectral_conv2d(
            hidden_channels * 4,
            hidden_channels * 4,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.eq_norm = nn.GroupNorm(8, hidden_channels * 4)

        # Output classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels * 4, 10)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Handle spectral norm wrapper
                if hasattr(m, "parametrizations"):
                    weight = m.parametrizations.weight.original
                else:
                    weight = m.weight
                nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="tanh")
                # Scale down for stability
                weight.data.mul_(0.5)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize hidden state for equilibrium block.
        Dimensions depend on input x (batch size) and stage 3 output spatial dim (8x8 for CIFAR).
        """
        # x is the raw input [B, C, H, W]
        B = x.shape[0]
        # Check input dimensions to avoid unpacking errors on flattened inputs
        if x.dim() == 4:
            H, W = x.shape[2], x.shape[3]
        else:
            # Fallback for flattened inputs if they occur (e.g., legacy tests)
            # Assume square image if flattened
            side = int((x.shape[1] / 3) ** 0.5)
            H, W = side, side

        # Assuming 3 downsampling stages with stride 1, 2, 2
        # Stage 1: stride 1 (32->32)
        # Stage 2: stride 2 (32->16)
        # Stage 3: stride 2 (16->8)
        # So H_out = H // 4, W_out = W // 4
        H_out, W_out = H // 4, W // 4
        return torch.zeros(
            B, self.hidden_dim, H_out, W_out, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Run feedforward stages to get input for equilibrium block."""
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        return h

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single step implementation (uncompiled)."""
        h_norm = self.eq_norm(h)

        if TritonEqPropOps.is_available() and h.is_cuda:
            # pre_act = eq_conv(h_norm) + x_transformed
            # fused update: lerp(h, tanh(pre_act), gamma)
            # Triton step computes: (1-alpha)h + alpha*tanh(pre_act)
            # This is exactly lerp(h, tanh(pre_act), alpha)
            pre_act = self.eq_conv(h_norm) + x_transformed
            return TritonEqPropOps.step(h, pre_act, alpha=self.gamma)

        # Add input drive to recurrent input
        h_next = torch.tanh(self.eq_conv(h_norm) + x_transformed)
        # Exponential moving average update
        return torch.lerp(h, h_next, self.gamma)

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Single equilibrium step.
        Injects the transformed input (stage3 features) into the recurrent dynamics.
        h_next = tanh(W * norm(h) + x_transformed)
        """
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        features = self.pool(h).flatten(1)
        return self.fc(features)

    def get_hebbian_pairs(self, h, x):
        """
        Hebbian updates for ModernConvEqProp.
        Only the recurrent layer (eq_conv) uses Hebbian learning in the equilibrium block.
        The feedforward stages (stage1, stage2, stage3) are trained via gradients backpropagated
        from the equilibrium block (which acts as a 'target generator').

        To enable learning in the feedforward stages via the contrastive rule, we need to capture
        the interaction between the stages' output (`x_transformed`) and the equilibrium state `h`.
        The stages effectively act as a complex layer mapping input `x` to `x_transformed`, which
        then drives `h`.

        By treating the entire feedforward pipeline as a single "layer" that outputs `x_transformed`,
        and providing `h` as the target, the `contrastive_update` mechanism will compute:
        `stage_pipeline(x) * h` (dot product).

        This correctly captures the interaction term `x_transformed * h`, allowing gradients to
        backpropagate through the stages based on the difference between the nudged and free phases.

        Returns:
           - Recurrent weights: `(eq_conv, h_norm, h)`
           - Feedforward weights: `(feedforward_container, x, h)`
        """
        # We need a container for the stages
        if not hasattr(self, "feedforward_net"):
            self.feedforward_net = nn.Sequential(self.stage1, self.stage2, self.stage3)

        h_norm = self.eq_norm(h)

        return [(self.eq_conv, h_norm, h), (self.feedforward_net, x, h)]


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
        # Single-stage embedding
        self.embed = spectral_conv2d(
            self.input_channels_count,
            self.hidden_channels,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        # Recurrent block
        self.W_rec = spectral_conv2d(
            self.hidden_channels,
            self.hidden_channels,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.norm = nn.GroupNorm(8, self.hidden_channels)

        # Classifier / Output Head
        if self.pool_output:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_channels, self.output_dim_val),
            )
        else:
            # Spatial output (e.g. for diffusion)
            self.head = spectral_conv2d(
                self.hidden_channels,
                self.output_dim_val,
                kernel_size=1,  # 1x1 conv for projection
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
