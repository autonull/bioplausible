"""
EquiTile Builder: Fluent API for Model Construction
====================================================

Provides a builder pattern for constructing EquiTile models
with a fluent, chainable API.

Examples
--------
>>> from bioplausible.equitile.builder import EquiTileBuilder
>>> model = (EquiTileBuilder()
...     .with_architecture(neurons_per_tile=64, tiles_per_layer=4, num_layers=4)
...     .with_io(input_dim=784, output_dim=10)
...     .with_learning_rate(0.01)
...     .with_mode('pc')
...     .build())

>>> # Research configuration
>>> model = (EquiTileBuilder.research()
...     .with_architecture(neurons_per_tile=64, tiles_per_layer=4)
...     .with_io(784, 10)
...     .enable_layer_norm()
...     .enable_curriculum()
...     .build())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import torch

if TYPE_CHECKING:
    from .core import EquiTile


@dataclass
class ArchitectureConfig:
    """Architecture configuration for builder.

    Attributes
    ----------
    neurons_per_tile : int
        Neurons per tile
    tiles_per_layer : int
        Tiles per layer
    num_layers : int
        Total number of layers
    """

    neurons_per_tile: int = 64
    tiles_per_layer: int = 4
    num_layers: int = 4


@dataclass
class IOConfig:
    """I/O configuration for builder.

    Attributes
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    """

    input_dim: int = 784
    output_dim: int = 10


@dataclass
class LearningConfig:
    """Learning configuration for builder.

    Attributes
    ----------
    learning_rate : float
        Base learning rate
    importance_lr : float
        Importance learning rate
    inference_steps : int
        Number of inference steps
    dropout : float
        Dropout probability
    weight_decay : float
        Weight decay
    """

    learning_rate: float = 0.01
    importance_lr: float = 0.001
    inference_steps: int = 10
    dropout: float = 0.1
    weight_decay: float = 1e-4


@dataclass
class DynamicsConfig:
    """Dynamics configuration for builder.

    Attributes
    ----------
    step_size : float
        Integration step size
    lambda_error : float
        Error weight
    beta : float
        Nudge strength (EP mode)
    clamp_activities : bool
        Clamp activities
    """

    step_size: float = 0.1
    lambda_error: float = 0.1
    beta: float = 0.1
    clamp_activities: bool = True


class EquiTileBuilder:
    """Fluent builder for EquiTile models.

    Provides a chainable API for constructing EquiTile models
    with various configurations.

    Examples
    --------
    Basic usage:
    >>> model = (EquiTileBuilder()
    ...     .with_architecture(neurons_per_tile=64, tiles_per_layer=4)
    ...     .with_io(input_dim=784, output_dim=10)
    ...     .with_learning_rate(0.01)
    ...     .build())

    Production configuration:
    >>> model = EquiTileBuilder.production(
    ...     input_dim=784,
    ...     output_dim=10,
    ... )

    Research configuration:
    >>> model = EquiTileBuilder.research(
    ...     input_dim=784,
    ...     output_dim=10,
    ... ).enable_layer_norm().enable_curriculum().build()
    """

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self._arch = ArchitectureConfig()
        self._io = IOConfig()
        self._learning = LearningConfig()
        self._dynamics = DynamicsConfig()
        self._mode: Literal["pc", "ep"] = "pc"
        self._activation: Literal["tanh", "relu", "gelu"] = "gelu"
        self._task_type: Literal[
            "classification", "regression", "binary", "multilabel"
        ] = "classification"
        self._gradient_clip: float = 1.0
        self._extra_kwargs: Dict[str, Any] = {}

    @classmethod
    def production(cls, input_dim: int = 784, output_dim: int = 10) -> EquiTileBuilder:
        """Create a production-ready builder.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension

        Returns
        -------
        EquiTileBuilder
            Configured builder
        """
        builder = cls()
        builder._arch = ArchitectureConfig(
            neurons_per_tile=64,
            tiles_per_layer=4,
            num_layers=4,
        )
        builder._io = IOConfig(input_dim=input_dim, output_dim=output_dim)
        builder._learning = LearningConfig(
            learning_rate=0.01,
            dropout=0.1,
            weight_decay=1e-4,
        )
        builder._mode = "pc"
        return builder

    @classmethod
    def research(cls, input_dim: int = 784, output_dim: int = 10) -> EquiTileBuilder:
        """Create a research-oriented builder.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension

        Returns
        -------
        EquiTileBuilder
            Configured builder
        """
        builder = cls()
        builder._arch = ArchitectureConfig(
            neurons_per_tile=64,
            tiles_per_layer=4,
            num_layers=4,
        )
        builder._io = IOConfig(input_dim=input_dim, output_dim=output_dim)
        builder._learning = LearningConfig(
            learning_rate=0.01,
            dropout=0.0,
            inference_steps=15,
        )
        builder._dynamics = DynamicsConfig(
            beta=0.1,
            step_size=0.1,
        )
        builder._mode = "ep"
        return builder

    @classmethod
    def fast(cls, input_dim: int = 784, output_dim: int = 10) -> EquiTileBuilder:
        """Create a fast prototyping builder.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension

        Returns
        -------
        EquiTileBuilder
            Configured builder
        """
        builder = cls()
        builder._arch = ArchitectureConfig(
            neurons_per_tile=32,
            tiles_per_layer=2,
            num_layers=3,
        )
        builder._io = IOConfig(input_dim=input_dim, output_dim=output_dim)
        builder._learning = LearningConfig(
            learning_rate=0.01,
            dropout=0.0,
            inference_steps=5,
        )
        builder._mode = "pc"
        return builder

    def with_architecture(
        self,
        neurons_per_tile: int,
        tiles_per_layer: int,
        num_layers: int,
    ) -> EquiTileBuilder:
        """Configure architecture.

        Parameters
        ----------
        neurons_per_tile : int
            Neurons per tile
        tiles_per_layer : int
            Tiles per layer
        num_layers : int
            Total layers

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._arch = ArchitectureConfig(
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_layers=num_layers,
        )
        return self

    def with_io(
        self,
        input_dim: int,
        output_dim: int,
    ) -> EquiTileBuilder:
        """Configure I/O dimensions.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._io = IOConfig(input_dim=input_dim, output_dim=output_dim)
        return self

    def with_learning_rate(
        self,
        learning_rate: float,
        importance_lr: Optional[float] = None,
    ) -> EquiTileBuilder:
        """Configure learning rate.

        Parameters
        ----------
        learning_rate : float
            Base learning rate
        importance_lr : float, optional
            Importance learning rate

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._learning.learning_rate = learning_rate
        if importance_lr is not None:
            self._learning.importance_lr = importance_lr
        return self

    def with_inference_steps(self, steps: int) -> EquiTileBuilder:
        """Configure inference steps.

        Parameters
        ----------
        steps : int
            Number of steps

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._learning.inference_steps = steps
        return self

    def with_dropout(self, dropout: float) -> EquiTileBuilder:
        """Configure dropout.

        Parameters
        ----------
        dropout : float
            Dropout probability

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._learning.dropout = dropout
        return self

    def with_weight_decay(self, weight_decay: float) -> EquiTileBuilder:
        """Configure weight decay.

        Parameters
        ----------
        weight_decay : float
            Weight decay

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._learning.weight_decay = weight_decay
        return self

    def with_mode(self, mode: Literal["pc", "ep"]) -> EquiTileBuilder:
        """Configure learning mode.

        Parameters
        ----------
        mode : str
            'pc' or 'ep'

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._mode = mode
        return self

    def with_activation(
        self,
        activation: Literal["tanh", "relu", "gelu"],
    ) -> EquiTileBuilder:
        """Configure activation function.

        Parameters
        ----------
        activation : str
            Activation function name

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._activation = activation
        return self

    def with_task_type(
        self,
        task_type: Literal["classification", "regression", "binary", "multilabel"],
    ) -> EquiTileBuilder:
        """Configure task type.

        Parameters
        ----------
        task_type : str
            Task type

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._task_type = task_type
        return self

    def with_dynamics(
        self,
        step_size: Optional[float] = None,
        lambda_error: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> EquiTileBuilder:
        """Configure dynamics parameters.

        Parameters
        ----------
        step_size : float, optional
            Integration step size
        lambda_error : float, optional
            Error weight
        beta : float, optional
            Nudge strength

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        if step_size is not None:
            self._dynamics.step_size = step_size
        if lambda_error is not None:
            self._dynamics.lambda_error = lambda_error
        if beta is not None:
            self._dynamics.beta = beta
        return self

    def with_gradient_clip(self, gradient_clip: float) -> EquiTileBuilder:
        """Configure gradient clipping.

        Parameters
        ----------
        gradient_clip : float
            Gradient clip value

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._gradient_clip = gradient_clip
        return self

    def with_sparsity(
        self,
        threshold: float,
        penalty: Optional[float] = None,
    ) -> EquiTileBuilder:
        """Configure sparsity settings.

        Parameters
        ----------
        threshold : float
            Activity threshold for sparsity
        penalty : float, optional
            Sparsity penalty coefficient

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._extra_kwargs["sparsity_threshold"] = threshold
        if penalty is not None:
            self._extra_kwargs["sparsity_penalty_coef"] = penalty
        return self

    def with_importance_learning(
        self,
        lr: float,
        decay: Optional[float] = None,
    ) -> EquiTileBuilder:
        """Configure importance learning settings.

        Parameters
        ----------
        lr : float
            Importance learning rate
        decay : float, optional
            Importance decay

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._learning.importance_lr = lr
        if decay is not None:
            self._extra_kwargs["importance_decay"] = decay
        return self

    def enable_clamping(self, enabled: bool = True) -> EquiTileBuilder:
        """Enable/disable activity clamping.

        Parameters
        ----------
        enabled : bool
            Whether to enable clamping

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._dynamics.clamp_activities = enabled
        return self

    def with_kwargs(self, **kwargs: Any) -> EquiTileBuilder:
        """Add extra keyword arguments.

        Parameters
        ----------
        **kwargs
            Extra arguments to pass to EquiTile

        Returns
        -------
        EquiTileBuilder
            Self for chaining
        """
        self._extra_kwargs.update(kwargs)
        return self

    def build(self) -> EquiTile:
        """Build the EquiTile model.

        Returns
        -------
        EquiTile
            Constructed model
        """
        from .config import EquiTileConfig
        from .core import EquiTile

        config = EquiTileConfig(
            neurons_per_tile=self._arch.neurons_per_tile,
            num_layers=self._arch.num_layers,
            tiles_per_layer=self._arch.tiles_per_layer,
            mode=self._mode,
            learning_rate=self._learning.learning_rate,
            importance_lr=self._learning.importance_lr,
            inference_steps=self._learning.inference_steps,
            step_size=self._dynamics.step_size,
            lambda_error=self._dynamics.lambda_error,
            beta=self._dynamics.beta,
            dropout=self._learning.dropout,
            weight_decay=self._learning.weight_decay,
            gradient_clip=self._gradient_clip,
            activation=self._activation,
            task_type=self._task_type,
            clamp_activities=self._dynamics.clamp_activities,
            **self._extra_kwargs,
        )

        return EquiTile(
            config=config,
            input_dim=self._io.input_dim,
            output_dim=self._io.output_dim,
        )


# =============================================================================
# Enhanced Builder
# =============================================================================


class EnhancedEquiTileBuilder(EquiTileBuilder):
    """Builder for Enhanced EquiTile with EP features.

    Extends EquiTileBuilder with additional EP-specific options.

    Examples
    --------
    >>> model = (EnhancedEquiTileBuilder()
    ...     .with_architecture(64, 4, 4)
    ...     .with_io(784, 10)
    ...     .enable_layer_norm()
    ...     .enable_curriculum(n_stages=5)
    ...     .build())
    """

    def __init__(self) -> None:
        """Initialize enhanced builder."""
        super().__init__()
        self._use_layer_norm: bool = False
        self._layer_norm_eps: float = 1e-5
        self._layer_norm_affine: bool = True
        self._use_curriculum: bool = False
        self._curriculum_stages: int = 5
        self._use_weight_norm: bool = False
        self._init_scheme: Literal["xavier", "kaiming", "orthogonal"] = "xavier"
        self._init_gain: float = 1.0

    def enable_layer_norm(
        self,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> EnhancedEquiTileBuilder:
        """Enable layer normalization.

        Parameters
        ----------
        eps : float
            Layer norm epsilon
        affine : bool
            Use affine parameters

        Returns
        -------
        EnhancedEquiTileBuilder
            Self for chaining
        """
        self._use_layer_norm = True
        self._layer_norm_eps = eps
        self._layer_norm_affine = affine
        return self

    def enable_curriculum(
        self,
        n_stages: int = 5,
    ) -> EnhancedEquiTileBuilder:
        """Enable curriculum learning.

        Parameters
        ----------
        n_stages : int
            Number of curriculum stages

        Returns
        -------
        EnhancedEquiTileBuilder
            Self for chaining
        """
        self._use_curriculum = True
        self._curriculum_stages = n_stages
        return self

    def enable_weight_norm(self) -> EnhancedEquiTileBuilder:
        """Enable weight normalization.

        Returns
        -------
        EnhancedEquiTileBuilder
            Self for chaining
        """
        self._use_weight_norm = True
        return self

    def with_initialization(
        self,
        scheme: Literal["xavier", "kaiming", "orthogonal"],
        gain: float = 1.0,
    ) -> EnhancedEquiTileBuilder:
        """Configure weight initialization.

        Parameters
        ----------
        scheme : str
            Initialization scheme
        gain : float
            Gain parameter

        Returns
        -------
        EnhancedEquiTileBuilder
            Self for chaining
        """
        self._init_scheme = scheme
        self._init_gain = gain
        return self

    def build(self):
        """Build the Enhanced EquiTile model.

        Returns
        -------
        EnhancedEquiTile
            Constructed enhanced model
        """
        from .enhanced import EnhancedEquiTile
        from .enhanced import EnhancedEquiTileConfig

        # Create enhanced config
        # Note: We combine base config parameters into the enhanced config
        config = EnhancedEquiTileConfig(
            # Base parameters
            neurons_per_tile=self._arch.neurons_per_tile,
            num_layers=self._arch.num_layers,
            tiles_per_layer=self._arch.tiles_per_layer,
            learning_rate=self._learning.learning_rate,
            importance_lr=self._learning.importance_lr,
            inference_steps=self._learning.inference_steps,
            dropout=self._learning.dropout,
            weight_decay=self._learning.weight_decay,
            step_size=self._dynamics.step_size,
            lambda_error=self._dynamics.lambda_error,
            beta=self._dynamics.beta,
            clamp_activities=self._dynamics.clamp_activities,
            mode=self._mode,
            gradient_clip=self._gradient_clip,
            # Enhanced parameters
            use_layer_norm=self._use_layer_norm,
            norm_eps=self._layer_norm_eps,
            use_curriculum=self._use_curriculum,
            curriculum_stages=self._curriculum_stages,
            # Kwargs mappings
            sparsity_threshold=self._extra_kwargs.get("sparsity_threshold", 0.01),
            importance_decay=self._extra_kwargs.get("importance_decay", 0.95),
        )

        # Instantiate EnhancedEquiTile directly
        return EnhancedEquiTile(
            neurons_per_tile=self._arch.neurons_per_tile,
            num_layers=self._arch.num_layers,
            tiles_per_layer=self._arch.tiles_per_layer,
            input_dim=self._io.input_dim,
            output_dim=self._io.output_dim,
            enhanced_config=config,
            activation=self._activation,
            task_type=self._task_type,
            **self._extra_kwargs,
        )


# =============================================================================
# Training Context Manager
# =============================================================================


class TrainingContext:
    """Context manager for training loops.

    Provides convenient training loop management with automatic
    logging, checkpointing, and early stopping.

    Parameters
    ----------
    model : EquiTile
        Model to train
    log_interval : int
        Log every N steps
    checkpoint_path : str, optional
        Path for checkpoints

    Examples
    --------
    >>> with TrainingContext(model, log_interval=100) as ctx:
    ...     for epoch in range(100):
    ...         for X, y in dataloader:
    ...             stats = ctx.train_step(X, y)
    ...         if ctx.should_checkpoint(epoch):
    ...             ctx.save_checkpoint(epoch)
    """

    def __init__(
        self,
        model: EquiTile,
        log_interval: int = 100,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.log_interval = log_interval
        self.checkpoint_path = checkpoint_path

        self._step_count = 0
        self._epoch = 0
        self._best_loss: float = float("inf")
        self._history: List[Dict[str, float]] = []

    def __enter__(self) -> TrainingContext:
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        pass

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform training step.

        Parameters
        ----------
        x : torch.Tensor
            Input
        y : torch.Tensor
            Target

        Returns
        -------
        dict
            Training statistics
        """
        stats = self.model.train_step(x, y)
        self._step_count += 1
        self._history.append(stats)

        if self._step_count % self.log_interval == 0:
            self._log_step(stats)

        return stats

    def _log_step(self, stats: Dict[str, float]) -> None:
        """Log training step.

        Parameters
        ----------
        stats : dict
            Training statistics
        """
        loss = stats.get("loss", 0.0)
        accuracy = stats.get("accuracy", 0.0)
        print(f"Step {self._step_count}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    def should_checkpoint(self, epoch: int) -> bool:
        """Check if should save checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch

        Returns
        -------
        bool
            Whether to checkpoint
        """
        self._epoch = epoch
        recent_loss = sum(s.get("loss", 0.0) for s in self._history[-10:]) / 10
        if recent_loss < self._best_loss:
            self._best_loss = recent_loss
            return True
        return False

    def save_checkpoint(
        self, epoch: int, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        str
            Checkpoint path
        """
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path not set")

        if metadata is None:
            metadata = {}

        metadata["epoch"] = epoch
        metadata["best_loss"] = self._best_loss

        self.model.save_checkpoint(self.checkpoint_path, metadata=metadata)
        return self.checkpoint_path

    @property
    def step_count(self) -> int:
        """Get step count."""
        return self._step_count

    @property
    def epoch(self) -> int:
        """Get current epoch."""
        return self._epoch

    @property
    def best_loss(self) -> float:
        """Get best loss."""
        return self._best_loss


# =============================================================================
# Inference Context Manager
# =============================================================================


class InferenceContext:
    """Context manager for inference.

    Sets model to eval mode and disables gradient computation.

    Parameters
    ----------
    model : EquiTile
        Model for inference

    Examples
    --------
    >>> with InferenceContext(model) as ctx:
    ...     predictions = ctx.predict(X)
    """

    def __init__(self, model: EquiTile) -> None:
        self.model = model
        self._training: bool = False

    def __enter__(self) -> InferenceContext:
        """Enter inference context."""
        self._training = self.model.training
        self.model.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit inference context."""
        if self._training:
            self.model.train()

    def predict(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run inference.

        Parameters
        ----------
        x : torch.Tensor
            Input
        steps : int, optional
            Relaxation steps

        Returns
        -------
        torch.Tensor
            Output logits
        """
        with torch.no_grad():
            return self.model(x, steps=steps)

    def predict_proba(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run inference and return probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input
        steps : int, optional
            Relaxation steps

        Returns
        -------
        torch.Tensor
            Output probabilities
        """
        logits = self.predict(x, steps)
        return torch.softmax(logits, dim=-1)

    def predict_class(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run inference and return class predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input
        steps : int, optional
            Relaxation steps

        Returns
        -------
        torch.Tensor
            Class predictions
        """
        logits = self.predict(x, steps)
        return logits.argmax(dim=-1)


# =============================================================================
# Factory Functions
# =============================================================================


def build_model(
    input_dim: int = 784,
    output_dim: int = 10,
    preset: Literal["production", "research", "fast"] = "production",
    **kwargs: Any,
) -> EquiTile:
    """Build EquiTile model using preset.

    Parameters
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    preset : str
        Model preset
    **kwargs
        Additional arguments

    Returns
    -------
    EquiTile
        Built model
    """
    if preset == "production":
        builder = EquiTileBuilder.production(input_dim, output_dim)
    elif preset == "research":
        builder = EquiTileBuilder.research(input_dim, output_dim)
    elif preset == "fast":
        builder = EquiTileBuilder.fast(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return builder.with_kwargs(**kwargs).build()


def build_enhanced_model(
    input_dim: int = 784,
    output_dim: int = 10,
    enable_layer_norm: bool = True,
    enable_curriculum: bool = False,
    **kwargs: Any,
) -> Any:
    """Build Enhanced EquiTile model.

    Parameters
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    enable_layer_norm : bool
        Enable layer normalization
    enable_curriculum : bool
        Enable curriculum learning
    **kwargs
        Additional arguments

    Returns
    -------
    EnhancedEquiTile
        Built enhanced model
    """
    builder = EnhancedEquiTileBuilder.research(input_dim, output_dim)

    if enable_layer_norm:
        builder.enable_layer_norm()
    if enable_curriculum:
        builder.enable_curriculum()

    return builder.with_kwargs(**kwargs).build()
