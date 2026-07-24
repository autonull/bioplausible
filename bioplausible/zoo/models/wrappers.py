"""
EqProp Model Wrappers

Generic wrappers that convert standard PyTorch modules into EqProp-compatible models.
This allows using PyTorch primitives while supporting equilibrium propagation.

Key insight: Any recurrent model that satisfies L < 1 can be trained with EqProp.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .base import EqPropModel


class RecurrentWrapper(EqPropModel):
    """
    Wrapper that converts any recurrent cell into an EqProp model.

    This is the generic version of LoopedMLP - it wraps nn.RNNCell,
    nn.LSTMCell, nn.GRUCell, or any custom recurrent cell.

    Example:
        # Using RNNCell
        cell = nn.RNNCell(input_dim, hidden_dim)
        model = RecurrentWrapper(cell, input_dim, hidden_dim, output_dim)

        # Using custom cell
        class MyCell(nn.Module):
            def forward(self, x, h):
                return torch.tanh(x @ self.W_x + h @ self.W_h)

        model = RecurrentWrapper(MyCell(), input_dim, hidden_dim, output_dim)
    """

    def __init__(
        self,
        cell: nn.Module,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
        )

        self.cell = cell

        # Apply spectral norm if requested
        if use_spectral_norm:
            self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        """Apply spectral normalization to cell weights."""
        for module in self.cell.modules():
            if isinstance(module, nn.Linear):
                spectral_norm(module)

    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Forward pass with settling dynamics."""
        steps = steps or self.max_steps
        batch_size = x.shape[0]

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Settle to equilibrium
        for _ in range(steps):
            h = self.cell(x, h)

        # Output from final hidden state
        return self.output_layer(h)


class StackedRecurrentWrapper(EqPropModel):
    """
    Stacked recurrent layers that settle to joint equilibrium.

    Multiple recurrent layers iterate together until the entire
    stack reaches a fixed point.

    Example:
        model = StackedRecurrentWrapper(
            cell_type='rnn',  # or 'lstm', 'gru'
            input_dim=784,
            hidden_dim=256,
            output_dim=10,
            num_layers=3,
        )
    """

    def __init__(
        self,
        cell_type: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
        )

        # Create stacked cells
        cell_class = {"rnn": nn.RNNCell, "lstm": nn.LSTMCell, "gru": nn.GRUCell}[
            cell_type
        ]

        self.cells = nn.ModuleList(
            [
                cell_class(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        # Apply spectral norm
        if use_spectral_norm:
            for cell in self.cells:
                for module in cell.modules():
                    if isinstance(module, nn.Linear):
                        spectral_norm(module)

        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Forward pass with joint settling dynamics."""
        steps = steps or self.max_steps
        batch_size = x.shape[0]

        # Initialize hidden states for all layers
        states = [
            torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]

        # Joint settling
        for _ in range(steps):
            # Layer 0
            h = self.cells[0](x, states[0])
            states[0] = h if isinstance(h, torch.Tensor) else h[0]

            # Layers 1..N
            for i in range(1, self.num_layers):
                h = self.cells[i](states[i - 1], states[i])
                states[i] = h if isinstance(h, torch.Tensor) else h[0]

        # Output from final layer
        return self.output_layer(states[-1])


class TransformerEqPropWrapper(EqPropModel):
    """
    Wrapper that converts PyTorch Transformer into EqProp model.

    The transformer encoder layers iterate together to equilibrium
    instead of feedforward processing.

    Example:
        model = TransformerEqPropWrapper(
            input_dim=784,
            hidden_dim=256,
            output_dim=10,
            num_heads=8,
            num_layers=4,
        )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
        )

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Apply spectral norm
        if use_spectral_norm:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    spectral_norm(module)

        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Forward pass with equilibrium dynamics."""
        steps = steps or self.max_steps

        # Project to hidden dim and add sequence dimension
        h = self.input_projection(x).unsqueeze(1)  # [batch, 1, hidden]

        # Settle to equilibrium (all layers iterate together)
        for _ in range(steps):
            h = self.transformer(h)

        # Output from final state
        return self.output_layer(h.squeeze(1))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_rnn_eqprop(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    cell_type: str = "rnn",
    num_layers: int = 1,
    **kwargs,
) -> EqPropModel:
    """Create EqProp RNN model."""
    if num_layers == 1:
        cell_class = {"rnn": nn.RNNCell, "lstm": nn.LSTMCell, "gru": nn.GRUCell}[
            cell_type
        ]
        cell = cell_class(input_dim, hidden_dim)
        return RecurrentWrapper(cell, input_dim, hidden_dim, output_dim, **kwargs)
    else:
        return StackedRecurrentWrapper(
            cell_type=cell_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            **kwargs,
        )


def create_transformer_eqprop(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_heads: int = 8,
    num_layers: int = 4,
    **kwargs,
) -> EqPropModel:
    """Create EqProp Transformer model."""
    return TransformerEqPropWrapper(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        **kwargs,
    )


__all__ = [
    "RecurrentWrapper",
    "StackedRecurrentWrapper",
    "TransformerEqPropWrapper",
    "create_rnn_eqprop",
    "create_transformer_eqprop",
]
