"""
EquiTile Configuration Classes
==============================

Consolidated configuration for all EquiTile components.
"""

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

# =============================================================================
# Sub-Configurations (Structured)
# =============================================================================


@dataclass
class ArchitectureConfig:
    """Architecture hyperparameters."""

    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 4


@dataclass
class OptimizationConfig:
    """Optimization hyperparameters."""

    learning_rate: float = 0.01
    importance_lr: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    dropout: float = 0.1
    importance_decay: float = 0.95
    importance_reg_coef: float = 0.01
    sparsity_penalty_coef: float = 0.05
    sparsity_threshold: float = 0.01
    min_active_fraction: float = 0.1


@dataclass
class DynamicsConfig:
    """Dynamics and Inference hyperparameters."""

    mode: Literal["pc", "ep", "backprop"] = "pc"
    inference_steps: int = 10
    step_size: float = 0.1
    lambda_error: float = 0.1
    beta: float = 0.1
    beta_anneal: float = 1.0
    inference_steps_free: Optional[int] = None
    inference_steps_nudged: Optional[int] = None
    use_symmetric_weights: bool = False
    clamp_activities: bool = True
    activity_clamp_min: float = -5.0
    activity_clamp_max: float = 5.0
    ep_init_scale: float = 0.1
    relaxation_tolerance: float = 1e-4


# =============================================================================
# Core Configuration
# =============================================================================


@dataclass
class EquiTileConfig:
    """Main EquiTile configuration.

    This class aggregates Architecture, Optimization, and Dynamics configurations.
    Fields are kept flat for backward compatibility and ease of use in CLI/Hyperopt.

    Architecture
    ------------
    neurons_per_tile: Number of neurons per tile (64-256 typical)
    num_layers: Total layers (input + hidden + output)
    tiles_per_layer: Tiles per hidden layer

    Learning & Regularization
    -------------------------
    learning_rate: Base learning rate for weight updates
    importance_lr: Learning rate for tile importance weights
    weight_decay: L2 regularization strength
    gradient_clip: Gradient clipping threshold (0 = disabled)
    dropout: Dropout probability (0 = disabled)
    importance_decay: EMA decay for importance tracking
    importance_reg_coef: Regularization coefficient for importance
    sparsity_penalty_coef: Penalty for non-sparse importance
    sparsity_threshold: Threshold for considering a tile "active"
    min_active_fraction: Minimum fraction of active tiles

    Dynamics & Mode
    ---------------
    mode: 'pc' (predictive coding), 'ep' (equilibrium propagation), or 'backprop'
    inference_steps: Number of relaxation steps during inference
    step_size: Integration step size for relaxation
    lambda_error: Weight of prediction error term in energy
    beta: Nudge strength for EP mode
    beta_anneal: Beta decay factor per step/epoch
    inference_steps_free: Separate steps for free phase (EP)
    inference_steps_nudged: Separate steps for nudged phase (EP)
    use_symmetric_weights: Enforce symmetric weights (for strict energy function)
    clamp_activities: Whether to clamp neuron activities
    activity_clamp_min: Minimum activity value
    activity_clamp_max: Maximum activity value
    ep_init_scale: Initialization scale for EP activities
    relaxation_tolerance: Tolerance for early stopping relaxation
    task_type: Literal["classification", "regression", "binary", "multilabel"] = "classification"
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    """

    # Architecture
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 4

    # Learning
    learning_rate: float = 0.01
    importance_lr: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    dropout: float = 0.1
    importance_decay: float = 0.95
    importance_reg_coef: float = 0.01
    sparsity_penalty_coef: float = 0.05
    sparsity_threshold: float = 0.01
    min_active_fraction: float = 0.1

    # Dynamics
    mode: Literal["pc", "ep", "backprop"] = "pc"
    inference_steps: int = 10
    step_size: float = 0.1
    lambda_error: float = 0.1
    beta: float = 0.1
    beta_anneal: float = 1.0
    inference_steps_free: Optional[int] = None
    inference_steps_nudged: Optional[int] = None
    use_symmetric_weights: bool = False
    clamp_activities: bool = True
    activity_clamp_min: float = -5.0
    activity_clamp_max: float = 5.0
    ep_init_scale: float = 0.1
    relaxation_tolerance: float = 1e-4

    # Task & Activation
    task_type: Literal["classification", "regression", "binary", "multilabel"] = (
        "classification"
    )
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration parameters."""
        if self.neurons_per_tile <= 0:
            raise ValueError(
                f"neurons_per_tile must be positive, got {self.neurons_per_tile}"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.tiles_per_layer <= 0:
            raise ValueError(
                f"tiles_per_layer must be positive, got {self.tiles_per_layer}"
            )

        if self.learning_rate < 0:
            raise ValueError(
                f"learning_rate must be non-negative, got {self.learning_rate}"
            )
        if self.importance_lr < 0:
            raise ValueError(
                f"importance_lr must be non-negative, got {self.importance_lr}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if not (0 <= self.sparsity_threshold <= 1):
            raise ValueError(
                f"sparsity_threshold must be in [0, 1], got {self.sparsity_threshold}"
            )
        if not (0 <= self.importance_decay <= 1):
            raise ValueError(
                f"importance_decay must be in [0, 1], got {self.importance_decay}"
            )

        if self.inference_steps < 0:
            raise ValueError(
                f"inference_steps must be non-negative, got {self.inference_steps}"
            )
        if self.mode not in ("pc", "ep", "backprop"):
            raise ValueError(
                f"Invalid mode {self.mode}, must be one of 'pc', 'ep', 'backprop'"
            )

    def to_architecture_config(self) -> ArchitectureConfig:
        return ArchitectureConfig(
            neurons_per_tile=self.neurons_per_tile,
            num_layers=self.num_layers,
            tiles_per_layer=self.tiles_per_layer,
        )

    def to_optimization_config(self) -> OptimizationConfig:
        return OptimizationConfig(
            learning_rate=self.learning_rate,
            importance_lr=self.importance_lr,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            dropout=self.dropout,
            importance_decay=self.importance_decay,
            importance_reg_coef=self.importance_reg_coef,
            sparsity_penalty_coef=self.sparsity_penalty_coef,
            sparsity_threshold=self.sparsity_threshold,
            min_active_fraction=self.min_active_fraction,
        )

    def to_dynamics_config(self) -> DynamicsConfig:
        return DynamicsConfig(
            mode=self.mode,
            inference_steps=self.inference_steps,
            step_size=self.step_size,
            lambda_error=self.lambda_error,
            beta=self.beta,
            beta_anneal=self.beta_anneal,
            inference_steps_free=self.inference_steps_free,
            inference_steps_nudged=self.inference_steps_nudged,
            use_symmetric_weights=self.use_symmetric_weights,
            clamp_activities=self.clamp_activities,
            activity_clamp_min=self.activity_clamp_min,
            activity_clamp_max=self.activity_clamp_max,
            ep_init_scale=self.ep_init_scale,
            relaxation_tolerance=self.relaxation_tolerance,
        )


@dataclass
class EnhancedEquiTileConfig(EquiTileConfig):
    """
    Enhanced configuration for EquiTile with all improvements.
    Inherits from EquiTileConfig for compatibility and reduced duplication.
    """

    # Normalization
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    norm_eps: float = 1e-6

    # Error Propagation
    use_residual_errors: bool = True
    residual_error_weight: float = 0.1
    use_error_momentum: bool = False
    error_momentum: float = 0.9

    # Learning Rate Adaptation
    per_tile_lr: bool = True
    lr_adaptation_rate: float = 0.01
    lr_adaptation_decay: float = 0.99
    min_lr_ratio: float = 0.1
    max_lr_ratio: float = 10.0

    # Momentum for Weight Updates
    use_weight_momentum: bool = True
    weight_momentum: float = 0.9

    # Weight Initialization
    deep_init: bool = True
    init_scale_factor: float = 1.0

    # Architecture Improvements
    use_skip_connections: bool = True
    skip_connection_weight: float = 0.5

    # Enhanced Tile Importance
    enhanced_importance: bool = True
    importance_competition: bool = True
    importance_entropy_weight: float = 0.01

    # Activity Improvements
    use_activity_clipping: bool = True
    activity_clip_value: float = 5.0
    use_activity_scaling: bool = False

    # Gradient Improvements
    use_gradient_centralization: bool = False

    # Curriculum Learning
    use_curriculum: bool = False
    curriculum_stages: int = 5

    # Monitoring
    track_tile_statistics: bool = True

    @classmethod
    def preset_minimal(cls) -> "EnhancedEquiTileConfig":
        """Minimal configuration (all improvements disabled)."""
        return cls(
            use_layer_norm=False,
            use_batch_norm=False,
            use_residual_errors=False,
            per_tile_lr=False,
            use_weight_momentum=False,
            deep_init=False,
            use_skip_connections=False,
            enhanced_importance=False,
            use_curriculum=False,
        )

    @classmethod
    def preset_vision(cls) -> "EnhancedEquiTileConfig":
        """Optimized for vision tasks (CNN-like behavior)."""
        return cls(
            use_layer_norm=True,
            use_batch_norm=True,
            use_residual_errors=True,
            per_tile_lr=True,
            use_weight_momentum=True,
            deep_init=True,
            use_skip_connections=True,
            enhanced_importance=True,
            use_gradient_centralization=True,
            dropout=0.2,
            use_curriculum=True,
        )

    @classmethod
    def preset_language(cls) -> "EnhancedEquiTileConfig":
        """Optimized for language modeling."""
        return cls(
            use_layer_norm=True,
            use_batch_norm=False,
            use_residual_errors=True,
            per_tile_lr=True,
            use_weight_momentum=True,
            deep_init=True,
            use_skip_connections=False,  # Skip connections can hurt language modeling
            enhanced_importance=True,
            dropout=0.1,
            use_curriculum=True,
        )

    @classmethod
    def preset_rl(cls) -> "EnhancedEquiTileConfig":
        """Optimized for reinforcement learning (CartPole, etc.)."""
        return cls(
            use_layer_norm=True,
            use_batch_norm=False,
            use_residual_errors=True,
            per_tile_lr=True,
            use_weight_momentum=True,
            deep_init=True,
            use_skip_connections=True,
            enhanced_importance=True,
            dropout=0.0,  # No dropout for RL
            use_curriculum=False,
        )


# =============================================================================
# Distributed Training Configuration
# =============================================================================


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    device_ids: List[int] = field(default_factory=list)
    tile_balance: Literal["round_robin", "layered", "balanced"] = "round_robin"
    communication_backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    mixed_precision_dtype: Literal["float16", "bfloat16"] = "float16"
    overlap_communication: bool = True
    sync_frequency: int = 1


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""

    device_ids: List[int] = field(default_factory=list)
    tile_assignment: Literal["round_robin", "layered", "balanced"] = "round_robin"
    sync_frequency: int = 1
    overlap_comm: bool = True
    async_execution: bool = True
    gradient_accumulation: int = 1


@dataclass
class NCCLConfig:
    """NCCL communication configuration."""

    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    backend: str = "nccl"
    timeout_minutes: int = 30
    init_method: str = "env://"


# =============================================================================
# Async Execution Configuration
# =============================================================================


@dataclass
class AsyncConfig:
    """Configuration for async tile execution."""

    n_workers: int = 4
    use_processes: bool = False
    device_ids: List[int] = field(default_factory=list)
    batch_threshold: int = 32
    priority_alpha: float = 0.5
    priority_beta: float = 0.5


# =============================================================================
# Enhanced EP Configuration
# =============================================================================


@dataclass
class EnhancedEPConfig:
    """Configuration for enhanced EP features.

    Deprecated: Use EnhancedEquiTileConfig instead.
    """

    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    layer_norm_affine: bool = True

    use_curriculum: bool = False
    curriculum_stages: int = 5

    use_weight_norm: bool = False

    init_scheme: Literal["xavier", "kaiming", "orthogonal"] = "xavier"
    init_gain: float = 1.0


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    enabled: bool = False
    curriculum_type: Literal["difficulty", "uncertainty", "loss"] = "difficulty"
    n_stages: int = 5
    samples_per_stage: int = 1000
    difficulty_metric: Literal["error", "loss", "uncertainty"] = "error"
    start_easy: bool = True
    auto_progress: bool = True
    progress_threshold: float = 0.1


# =============================================================================
# Tile Dynamics Configuration
# =============================================================================


@dataclass
class TileGrowthConfig:
    """Tile growth and pruning configuration."""

    # Growth
    growth_enabled: bool = True
    growth_threshold: float = 0.5
    growth_cooldown: int = 100
    max_tiles: int = 100
    max_tiles_per_layer: int = 16

    # Pruning
    prune_enabled: bool = True
    prune_threshold: float = 0.05
    prune_cooldown: int = 200
    min_tiles: int = 2
    min_tiles_per_layer: int = 1

    # Merging
    merge_enabled: bool = False
    merge_threshold: float = 0.8
    merge_cooldown: int = 500

    # Splitting
    split_enabled: bool = False
    split_threshold: float = 1.0
    split_cooldown: int = 300

    # General
    error_ema_decay: float = 0.95
    min_age_for_modify: int = 50


@dataclass
class DynamicEquiTileConfig:
    """Dynamic tile architecture configuration."""

    growth: TileGrowthConfig = field(default_factory=TileGrowthConfig)
    merge_enabled: bool = False
    split_enabled: bool = False
    track_history: bool = True
    max_history: int = 1000


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_production_config(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    **kwargs: Any,
) -> EquiTileConfig:
    """Create a production-ready configuration."""
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        mode="pc",
        dropout=0.1,
        weight_decay=1e-4,
        gradient_clip=1.0,
        **kwargs,
    )


def create_research_config(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    **kwargs: Any,
) -> EquiTileConfig:
    """Create a research configuration for EP studies."""
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        mode="ep",
        beta=0.1,
        beta_anneal=0.99,
        inference_steps_free=15,
        inference_steps_nudged=15,
        **kwargs,
    )


def create_fast_config(
    neurons_per_tile: int = 32,
    num_layers: int = 3,
    tiles_per_layer: int = 2,
    **kwargs: Any,
) -> EquiTileConfig:
    """Create a fast configuration for prototyping."""
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        inference_steps=5,
        dropout=0.0,
        **kwargs,
    )


def create_enhanced_config(
    use_layer_norm: bool = True,
    use_curriculum: bool = True,
    curriculum_stages: int = 5,
    **kwargs: Any,
) -> EnhancedEquiTileConfig:
    """Create enhanced EP configuration."""
    return EnhancedEquiTileConfig(
        use_layer_norm=use_layer_norm,
        use_curriculum=use_curriculum,
        curriculum_stages=curriculum_stages,
        **kwargs,
    )


def create_dynamic_config(
    growth_enabled: bool = True,
    prune_enabled: bool = True,
    **kwargs: Any,
) -> DynamicEquiTileConfig:
    """Create dynamic tile configuration."""
    return DynamicEquiTileConfig(
        growth=TileGrowthConfig(
            growth_enabled=growth_enabled, prune_enabled=prune_enabled, **kwargs
        ),
        **kwargs,
    )
