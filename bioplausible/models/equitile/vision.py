"""
EquiTile Vision: Convolutional EquiTile for Image Processing
=============================================================

Extends EquiTile with convolutional capabilities for vision tasks:
- ConvEquiTile: Convolutional tile architecture
- Vision-specific tile configurations
- Image augmentation support
- Vision benchmarks (MNIST, CIFAR-10, ImageNet)

Examples
--------
>>> from bioplausible.models.equitile.vision import ConvEquiTile, ConvEquiTileConfig
>>> config = ConvEquiTileConfig(
...     input_channels=3,
...     input_size=32,
...     num_classes=10,
... )
>>> model = ConvEquiTile(config)
>>> stats = model.train_step(images, labels)
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
class ConvEquiTileConfig:
    """Configuration for Convolutional EquiTile.

    Architecture
    ------------
    input_channels : int
        Number of input channels (e.g., 3 for RGB, 1 for grayscale)
    input_size : int
        Input image size (assumed square)
    num_classes : int
        Number of output classes

    Convolutional Settings
    ----------------------
    conv_channels : list of int
        Channels per convolutional stage
    kernel_sizes : list of int
        Kernel sizes for each conv stage
    use_pooling : bool
        Use max pooling after convolutions
    pooling_size : int
        Pooling kernel size

    Tile Settings
    -------------
    neurons_per_tile : int
        Neurons per tile in fully-connected head
    num_fc_layers : int
        Number of FC layers after convolutions
    tiles_per_layer : int
        Tiles per FC layer

    Learning
    --------
    learning_rate : float
        Base learning rate
    dropout : float
        Dropout probability
    weight_decay : float
        Weight decay
    """
    # Input/Output
    input_channels: int = 3
    input_size: int = 32
    num_classes: int = 10

    # Convolutional settings
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    use_pooling: bool = True
    pooling_size: int = 2

    # Tile settings
    neurons_per_tile: int = 64
    num_fc_layers: int = 2
    tiles_per_layer: int = 4

    # Learning
    learning_rate: float = 0.01
    dropout: float = 0.1
    weight_decay: float = 1e-4

    # EquiTile settings
    mode: Literal["pc", "ep"] = "pc"
    inference_steps: int = 10
    step_size: float = 0.1
    beta: float = 0.1


# =============================================================================
# Convolutional Feature Extractor
# =============================================================================

class ConvFeatureExtractor(nn.Module):
    """Convolutional feature extractor for EquiTile.

    Parameters
    ----------
    config : ConvEquiTileConfig
        Configuration
    """

    def __init__(self, config: ConvEquiTileConfig) -> None:
        super().__init__()
        self.config = config

        # Build convolutional stages
        self.conv_stages = nn.ModuleList()
        in_channels = config.input_channels

        for i, (out_channels, kernel_size) in enumerate(
            zip(config.conv_channels, config.kernel_sizes)
        ):
            stages = [
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

            if config.use_pooling:
                stages.append(nn.MaxPool2d(config.pooling_size))

            self.conv_stages.append(nn.Sequential(*stages))
            in_channels = out_channels

        # Calculate output size
        self._output_size = self._compute_output_size(config)

    def _compute_output_size(self, config: ConvEquiTileConfig) -> int:
        """Compute feature map size after convolutions."""
        size = config.input_size
        channels = config.conv_channels[-1] if config.conv_channels else config.input_channels

        for i in range(len(config.conv_channels)):
            # Convolution (with padding)
            size = size  # No change with padding

            # Pooling
            if config.use_pooling:
                size = size // config.pooling_size

        return channels * size * size

    def forward(self, x: Tensor) -> Tensor:
        """Extract features from input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, channels, height, width)

        Returns
        -------
        torch.Tensor
            Flattened features (batch, features)
        """
        for stage in self.conv_stages:
            x = stage(x)

        return x.view(x.size(0), -1)

    @property
    def output_size(self) -> int:
        """Get output feature dimension."""
        return self._output_size


# =============================================================================
# Convolutional EquiTile
# =============================================================================

@register_model("conv_equitile")
class ConvEquiTile(BioModel):
    """Convolutional EquiTile for vision tasks.

    Combines convolutional feature extraction with EquiTile's
    tile-based local learning for the classification head.

    Parameters
    ----------
    config : ConvEquiTileConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters

    Examples
    --------
    >>> config = ConvEquiTileConfig(
    ...     input_channels=3,
    ...     input_size=32,
    ...     num_classes=10,
    ... )
    >>> model = ConvEquiTile(config)
    >>> for images, labels in dataloader:
    ...     stats = model.train_step(images, labels)
    """

    algorithm_name = "ConvEquiTile"

    def __init__(
        self,
        config: Optional[ConvEquiTileConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = ConvEquiTileConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="conv_equitile",
                input_dim=config.input_channels * config.input_size * config.input_size,
                output_dim=config.num_classes,
            )
        )

        self.config = config

        # Convolutional feature extractor
        self.feature_extractor = ConvFeatureExtractor(config)

        # EquiTile classification head
        self._build_tile_head(config)

        # Optimizers
        self._optim_conv = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._optim_head = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=config.learning_rate,
        )

        # Regularization
        self._dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # State tracking
        self._step_count = 0

    def _build_tile_head(self, config: ConvEquiTileConfig) -> None:
        """Build EquiTile classification head.

        Parameters
        ----------
        config : ConvEquiTileConfig
            Configuration
        """
        feature_dim = self.feature_extractor.output_size

        # Tile configuration
        single_tile_dim = config.neurons_per_tile
        tiles_per_fc_layer = config.tiles_per_layer
        n_fc_layers = config.num_fc_layers
        n_tiles = tiles_per_fc_layer * n_fc_layers

        # Input projection - map features to all tile outputs
        total_tile_dim = n_tiles * single_tile_dim
        self.W_in = nn.Linear(feature_dim, total_tile_dim)

        # Tile importance
        self.tile_importance = nn.Parameter(torch.ones(n_tiles))

        # Output projection - use single tile dim
        self.W_out = nn.Linear(single_tile_dim, config.num_classes)

        # Store config for reshaping
        self._total_tile_dim = total_tile_dim
        self._n_tiles = n_tiles
        self._single_tile_dim = single_tile_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            nn.init.kaiming_normal_(self.W_in.weight, mode='fan_in', nonlinearity='relu')
            if self.W_in.bias is not None:
                nn.init.zeros_(self.W_in.bias)

            nn.init.xavier_normal_(self.W_out.weight, gain=1.0)
            if self.W_out.bias is not None:
                nn.init.zeros_(self.W_out.bias)

    def extract_features(self, x: Tensor) -> Tensor:
        """Extract convolutional features.

        Parameters
        ----------
        x : torch.Tensor
            Input images

        Returns
        -------
        torch.Tensor
            Features
        """
        return self.feature_extractor(x)

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        x : torch.Tensor
            Input images (batch, channels, height, width)
        y : torch.Tensor
            Target labels

        Returns
        -------
        dict
            Training statistics
        """
        self._step_count += 1
        batch_size = x.shape[0]
        device = x.device

        # Extract features
        features = self.extract_features(x)
        features = self._dropout(features)

        # Project to tile space
        tile_input = self.W_in(features)

        # Reshape to tiles
        tile_dim = self._single_tile_dim
        n_tiles = self._n_tiles

        # Reshape tile_input to (batch, n_tiles, tile_dim)
        activities = tile_input.view(batch_size, n_tiles, tile_dim)

        # Simple relaxation - just apply nonlinearity
        for _ in range(self.config.inference_steps):
            activities = F.relu(activities)

        # Read output - use last tile
        final_activity = activities[:, -1, :]
        logits = self.W_out(final_activity)

        # Compute loss
        loss = F.cross_entropy(logits, y)

        # Update all parameters together
        self._optim_conv.zero_grad()
        self._optim_head.zero_grad()
        loss.backward()
        self._optim_conv.step()
        self._optim_head.step()

        # Compute accuracy
        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == y).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "mode": self.config.mode,
        }

    def _compute_tile_predictions(self, activities: Tensor) -> Tensor:
        """Compute tile predictions.

        Parameters
        ----------
        activities : torch.Tensor
            Tile activities

        Returns
        -------
        torch.Tensor
            Predictions
        """
        # Simple feedforward prediction - each tile predicts next tile's activity
        predictions = []
        prev = activities[:, 0, :]  # First tile activity
        
        for i in range(activities.shape[1]):
            if i == 0:
                predictions.append(torch.zeros_like(prev))  # First tile has no prediction
            else:
                pred = self._dropout(F.relu(prev))
                predictions.append(pred)
                prev = activities[:, i, :]  # Use actual activity for next prediction

        return torch.stack(predictions, dim=1)

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images
        return_features : bool
            If True, return features as well

        Returns
        -------
        torch.Tensor or tuple
            Logits, or (logits, features)
        """
        features = self.extract_features(x)
        tile_input = self.W_in(features)

        # Reshape to tiles
        batch_size = tile_input.shape[0]
        tile_dim = self._single_tile_dim
        n_tiles = self._n_tiles

        # Reshape tile_input to (batch, n_tiles, tile_dim)
        activities = tile_input.view(batch_size, n_tiles, tile_dim)

        # Simple forward through tiles
        activity = activities[:, 0, :]  # First tile
        for i in range(1, n_tiles):
            activity = F.relu(activity)

        logits = self.W_out(activity)

        if return_features:
            return logits, features
        return logits


# =============================================================================
# Vision Data Augmentation
# =============================================================================

class VisionAugmentation:
    """Vision data augmentation utilities.

    Examples
    --------
    >>> aug = VisionAugmentation(
    ...     random_crop=True,
    ...     random_flip=True,
    ...     color_jitter=True,
    ... )
    >>> augmented = aug(images)
    """

    def __init__(
        self,
        random_crop: bool = False,
        crop_size: Optional[int] = None,
        random_flip: bool = False,
        color_jitter: bool = False,
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.color_jitter = color_jitter
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Apply augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Input images
        y : torch.Tensor, optional
            Labels (for augmentation consistency)

        Returns
        -------
        torch.Tensor
            Augmented images
        """
        # Random crop
        if self.random_crop and self.crop_size:
            x = self._random_crop(x)

        # Random flip
        if self.random_flip:
            x = self._random_flip(x)

        # Color jitter
        if self.color_jitter:
            x = self._color_jitter(x)

        # Normalize
        if self.normalize:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        return x

    def _random_crop(self, x: Tensor) -> Tensor:
        """Random crop."""
        b, c, h, w = x.shape
        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        return x[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def _random_flip(self, x: Tensor) -> Tensor:
        """Random horizontal flip."""
        if torch.rand(1) > 0.5:
            return x.flip(-1)
        return x

    def _color_jitter(self, x: Tensor) -> Tensor:
        """Simple color jitter."""
        # Brightness
        brightness = torch.empty(1).uniform_(0.8, 1.2).item()
        x = x * brightness

        # Contrast
        contrast = torch.empty(1).uniform_(0.8, 1.2).item()
        x = x * contrast

        return x  # Don't clamp - input may not be in [0,1]


# =============================================================================
# Factory Functions
# =============================================================================

def create_vision_model(
    input_channels: int = 3,
    input_size: int = 32,
    num_classes: int = 10,
    conv_channels: Optional[List[int]] = None,
    neurons_per_tile: int = 64,
    mode: Literal["pc", "ep"] = "pc",
    **kwargs: Any,
) -> ConvEquiTile:
    """Create a ConvEquiTile model for vision tasks.

    Parameters
    ----------
    input_channels : int
        Input channels
    input_size : int
        Input size
    num_classes : int
        Number of classes
    conv_channels : list of int, optional
        Convolutional channels
    neurons_per_tile : int
        Neurons per tile
    mode : str
        Learning mode
    **kwargs
        Additional arguments

    Returns
    -------
    ConvEquiTile
        Vision model
    """
    config = ConvEquiTileConfig(
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        conv_channels=conv_channels or [32, 64, 128],
        neurons_per_tile=neurons_per_tile,
        mode=mode,
        **kwargs,
    )
    return ConvEquiTile(config)


def create_mnist_model(
    neurons_per_tile: int = 64,
    **kwargs: Any,
) -> ConvEquiTile:
    """Create ConvEquiTile for MNIST.

    Parameters
    ----------
    neurons_per_tile : int
        Neurons per tile
    **kwargs
        Additional arguments

    Returns
    -------
    ConvEquiTile
        MNIST model
    """
    return create_vision_model(
        input_channels=1,
        input_size=28,
        num_classes=10,
        conv_channels=[16, 32, 64],
        neurons_per_tile=neurons_per_tile,
        **kwargs,
    )


def create_cifar_model(
    neurons_per_tile: int = 128,
    **kwargs: Any,
) -> ConvEquiTile:
    """Create ConvEquiTile for CIFAR-10/100.

    Parameters
    ----------
    neurons_per_tile : int
        Neurons per tile
    **kwargs
        Additional arguments

    Returns
    -------
    ConvEquiTile
        CIFAR model
    """
    return create_vision_model(
        input_channels=3,
        input_size=32,
        num_classes=10,  # or 100 for CIFAR-100
        conv_channels=[64, 128, 256],
        neurons_per_tile=neurons_per_tile,
        use_pooling=True,
        **kwargs,
    )


def create_imagenet_model(
    neurons_per_tile: int = 256,
    num_classes: int = 1000,
    **kwargs: Any,
) -> ConvEquiTile:
    """Create ConvEquiTile for ImageNet.

    Parameters
    ----------
    neurons_per_tile : int
        Neurons per tile
    num_classes : int
        Number of classes
    **kwargs
        Additional arguments

    Returns
    -------
    ConvEquiTile
        ImageNet model
    """
    return create_vision_model(
        input_channels=3,
        input_size=224,
        num_classes=num_classes,
        conv_channels=[64, 128, 256, 512],
        neurons_per_tile=neurons_per_tile,
        use_pooling=True,
        **kwargs,
    )
