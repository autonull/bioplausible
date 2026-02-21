"""
EquiTile Time Series: Sequential Data Modeling
===============================================

Extends EquiTile for time series and sequential data:
- TimeSeriesEquiTile: Recurrent and convolutional architectures
- Temporal attention mechanisms
- Support for forecasting, classification, and anomaly detection
- Multi-variate time series support

Examples
--------
>>> from bioplausible.models.equitile.timeseries import TimeSeriesEquiTile, TimeSeriesConfig
>>> config = TimeSeriesConfig(
...     input_dim=10,
...     seq_len=100,
...     output_dim=1,
...     model_type="forecasting",
... )
>>> model = TimeSeriesEquiTile(config)
>>> predictions = model(sequence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.models.base import BioModel, ModelConfig, register_model

if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TimeSeriesConfig:
    """Configuration for Time Series EquiTile.

    Input/Output
    ------------
    input_dim : int
        Input feature dimension
    seq_len : int
        Input sequence length
    output_dim : int
        Output dimension
    pred_len : int
        Prediction length (for forecasting)

    Architecture
    ------------
    model_type : str
        Model type: 'forecasting', 'classification', 'anomaly_detection'
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of layers
    neurons_per_tile : int
        Neurons per tile
    tiles_per_layer : int
        Tiles per layer

    Temporal Settings
    -----------------
    use_positional_encoding : bool
        Use positional encoding
    use_temporal_attention : bool
        Use temporal attention
    attention_heads : int
        Number of attention heads

    Learning
    --------
    learning_rate : float
        Base learning rate
    dropout : float
        Dropout probability
    """
    # Input/Output
    input_dim: int = 10
    seq_len: int = 100
    output_dim: int = 1
    pred_len: int = 10

    # Architecture
    model_type: Literal["forecasting", "classification", "anomaly_detection"] = "forecasting"
    hidden_dim: int = 64
    num_layers: int = 3
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4
    attention_heads: int = 4

    # Temporal settings
    use_positional_encoding: bool = True
    use_temporal_attention: bool = True

    # Learning
    learning_rate: float = 1e-3
    dropout: float = 0.1

    # EquiTile settings
    mode: Literal["pc", "ep"] = "pc"
    inference_steps: int = 5


# =============================================================================
# Positional Encoding
# =============================================================================

class TemporalPositionalEncoding(nn.Module):
    """Positional encoding for time series.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    max_len : int
        Maximum sequence length
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)

        Returns
        -------
        torch.Tensor
            Output with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# Temporal Attention
# =============================================================================

class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for time series.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)
        mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


# =============================================================================
# Time Series EquiTile Layer
# =============================================================================

class TimeSeriesEquiTileLayer(nn.Module):
    """Time Series EquiTile layer.

    Parameters
    ----------
    config : TimeSeriesConfig
        Configuration
    """

    def __init__(self, config: TimeSeriesConfig) -> None:
        super().__init__()
        self.config = config

        # Temporal attention
        if config.use_temporal_attention:
            self.attention = TemporalAttentionLayer(
                embed_dim=config.hidden_dim,
                num_heads=config.attention_heads,
                dropout=config.dropout,
            )
            self.norm1 = nn.LayerNorm(config.hidden_dim)
        else:
            self.attention = None
            self.norm1 = None

        # Layer norm
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Tile integration
        tile_dim = config.neurons_per_tile * config.tiles_per_layer
        self.tile_proj_in = nn.Linear(config.hidden_dim, tile_dim)
        self.tile_proj_out = nn.Linear(tile_dim, config.hidden_dim)
        self.tile_importance = nn.Parameter(torch.ones(config.tiles_per_layer))

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Temporal attention with residual
        if self.attention is not None:
            attn_output = self.attention(x, mask)
            x = x + attn_output
            x = self.norm1(x)

        # Tile-based processing
        tile_input = self.tile_proj_in(x)
        tile_output = self._process_tiles(tile_input)
        x = x + self.tile_proj_out(tile_output)

        # Feedforward with residual
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        return x

    def _process_tiles(self, x: Tensor) -> Tensor:
        """Process through tiles.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        tile_dim = self.config.neurons_per_tile
        n_tiles = self.config.tiles_per_layer

        # Reshape to tiles
        x = x.view(batch_size, seq_len, n_tiles, tile_dim)

        # Process each tile with importance weighting
        outputs = []
        for i in range(n_tiles):
            imp = torch.sigmoid(self.tile_importance[i])
            tile_out = F.relu(x[:, :, i, :]) * imp
            outputs.append(tile_out)

        return torch.stack(outputs, dim=2).view(batch_size, seq_len, -1)


# =============================================================================
# Time Series EquiTile
# =============================================================================

@register_model("timeseries_equitile")
class TimeSeriesEquiTile(BioModel):
    """Time Series EquiTile for sequential data.

    Combines temporal attention with EquiTile's tile-based
    processing for forecasting, classification, and anomaly detection.

    Parameters
    ----------
    config : TimeSeriesConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters

    Examples
    --------
    >>> config = TimeSeriesConfig(
    ...     input_dim=10,
    ...     seq_len=100,
    ...     output_dim=1,
    ...     model_type="forecasting",
    ... )
    >>> model = TimeSeriesEquiTile(config)
    >>> predictions = model(sequence)
    """

    algorithm_name = "TimeSeriesEquiTile"

    def __init__(
        self,
        config: Optional[TimeSeriesConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = TimeSeriesConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="timeseries_equitile",
                input_dim=config.input_dim,
                output_dim=config.output_dim,
            )
        )

        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = TemporalPositionalEncoding(
                embed_dim=config.hidden_dim,
                max_len=config.seq_len,
                dropout=config.dropout,
            )
        else:
            self.pos_encoding = None

        # Time series layers
        self.layers = nn.ModuleList([
            TimeSeriesEquiTileLayer(config) for _ in range(config.num_layers)
        ])

        # Output projection based on task
        if config.model_type == "forecasting":
            self.output_proj = nn.Linear(config.hidden_dim, config.pred_len * config.output_dim)
        elif config.model_type == "classification":
            self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        elif config.model_type == "anomaly_detection":
            self.output_proj = nn.Linear(config.hidden_dim, config.input_dim)
        else:
            self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, input_dim)
        mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        batch_size = x.shape[0]

        # Input projection
        x = self.input_proj(x)

        # Positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Time series layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection based on task
        if self.config.model_type == "forecasting":
            # Use last time step for forecasting
            x = x[:, -1, :]
            x = self.output_proj(x)
            x = x.view(batch_size, self.config.pred_len, self.config.output_dim)
        elif self.config.model_type == "classification":
            # Global average pooling for classification
            x = x.mean(dim=1)
            x = self.output_proj(x)
        elif self.config.model_type == "anomaly_detection":
            # Reconstruct input
            x = self.output_proj(x)
        else:
            x = self.output_proj(x)

        return x

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Target tensor
        mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        dict
            Training statistics
        """
        # Forward pass
        predictions = self.forward(x, mask)

        # Compute loss based on task
        if self.config.model_type == "forecasting":
            loss = F.mse_loss(predictions, y)
        elif self.config.model_type == "classification":
            loss = F.cross_entropy(predictions, y)
        elif self.config.model_type == "anomaly_detection":
            # Reconstruction loss
            loss = F.mse_loss(predictions, x)
        else:
            loss = F.mse_loss(predictions, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            if self.config.model_type == "forecasting":
                mae = F.l1_loss(predictions, y).item()
                metrics = {"mae": mae}
            elif self.config.model_type == "classification":
                accuracy = (predictions.argmax(dim=-1) == y).float().mean().item()
                metrics = {"accuracy": accuracy}
            else:
                metrics = {}

        return {
            "loss": loss.item(),
            **metrics,
        }

    def forecast(
        self,
        x: Tensor,
        steps: Optional[int] = None,
    ) -> Tensor:
        """Make forecasts.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence
        steps : int, optional
            Number of steps to forecast

        Returns
        -------
        torch.Tensor
            Forecasts
        """
        self.eval()
        with torch.no_grad():
            if steps is not None:
                # Auto-regressive forecasting
                predictions = []
                current_input = x.clone()

                for _ in range(steps):
                    pred = self.forward(current_input)
                    if pred.dim() > 2:
                        pred = pred[:, -1:, :]
                    predictions.append(pred)

                    # Update input
                    if current_input.shape[1] >= self.config.seq_len:
                        current_input = torch.cat([current_input[:, 1:, :], pred], dim=1)
                    else:
                        current_input = torch.cat([current_input, pred], dim=1)

                return torch.cat(predictions, dim=1)
            else:
                return self.forward(x)

    def detect_anomalies(
        self,
        x: Tensor,
        threshold: float = 0.1,
    ) -> Tuple[Tensor, Tensor]:
        """Detect anomalies using reconstruction error.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence
        threshold : float
            Anomaly threshold

        Returns
        -------
        tuple
            (anomaly_scores, anomaly_flags)
        """
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            error = torch.abs(x - reconstruction).mean(dim=-1)
            anomaly_flags = error > threshold
            return error, anomaly_flags


# =============================================================================
# Factory Functions
# =============================================================================

def create_forecasting_model(
    input_dim: int,
    seq_len: int,
    pred_len: int,
    **kwargs: Any,
) -> TimeSeriesEquiTile:
    """Create forecasting model.

    Parameters
    ----------
    input_dim : int
        Input dimension
    seq_len : int
        Sequence length
    pred_len : int
        Prediction length
    **kwargs
        Additional arguments

    Returns
    -------
    TimeSeriesEquiTile
        Forecasting model
    """
    config = TimeSeriesConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        output_dim=input_dim,
        pred_len=pred_len,
        model_type="forecasting",
        **kwargs,
    )
    return TimeSeriesEquiTile(config)


def create_classification_model(
    input_dim: int,
    seq_len: int,
    num_classes: int,
    **kwargs: Any,
) -> TimeSeriesEquiTile:
    """Create classification model.

    Parameters
    ----------
    input_dim : int
        Input dimension
    seq_len : int
        Sequence length
    num_classes : int
        Number of classes
    **kwargs
        Additional arguments

    Returns
    -------
    TimeSeriesEquiTile
        Classification model
    """
    config = TimeSeriesConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        output_dim=num_classes,
        model_type="classification",
        **kwargs,
    )
    return TimeSeriesEquiTile(config)


def create_anomaly_detection_model(
    input_dim: int,
    seq_len: int,
    **kwargs: Any,
) -> TimeSeriesEquiTile:
    """Create anomaly detection model.

    Parameters
    ----------
    input_dim : int
        Input dimension
    seq_len : int
        Sequence length
    **kwargs
        Additional arguments

    Returns
    -------
    TimeSeriesEquiTile
        Anomaly detection model
    """
    config = TimeSeriesConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        output_dim=input_dim,
        model_type="anomaly_detection",
        **kwargs,
    )
    return TimeSeriesEquiTile(config)
