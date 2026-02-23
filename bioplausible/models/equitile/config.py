"""
EquiTile Configuration Classes
==============================

Consolidated configuration for all EquiTile components.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# =============================================================================
# Core Configuration
# =============================================================================

@dataclass
class EquiTileConfig:
    """Main EquiTile configuration.

    Architecture
    ------------
    neurons_per_tile: Number of neurons per tile (64-256 typical)
    num_layers: Total layers (input + hidden + output)
    tiles_per_layer: Tiles per hidden layer

    Learning
    --------
    learning_rate: Base learning rate for weight updates
    importance_lr: Learning rate for tile importance weights
    inference_steps: Number of relaxation steps during inference

    Dynamics
    --------
    step_size: Integration step size for relaxation
    lambda_error: Weight of prediction error term in energy
    beta: Nudge strength for EP mode

    Regularization
    --------------
    dropout: Dropout probability (0 = disabled)
    weight_decay: L2 regularization strength
    gradient_clip: Gradient clipping threshold (0 = disabled)
    importance_decay: EMA decay for importance tracking

    Mode
    ----
    mode: 'pc' (predictive coding) or 'ep' (equilibrium propagation)
    """
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 4

    # Learning
    learning_rate: float = 0.01
    importance_lr: float = 0.001
    inference_steps: int = 10

    # Dynamics
    step_size: float = 0.1
    lambda_error: float = 0.1
    beta: float = 0.1

    # EP-specific
    inference_steps_free: Optional[int] = None
    inference_steps_nudged: Optional[int] = None
    beta_anneal: float = 1.0
    use_symmetric_weights: bool = False

    # Adaptive computation
    sparsity_threshold: float = 0.01
    min_active_fraction: float = 0.1

    # Regularization
    dropout: float = 0.1
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    importance_decay: float = 0.95

    # Behavior
    mode: Literal["pc", "ep"] = "pc"
    clamp_activities: bool = True
    relaxation_tolerance: float = 1e-4


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
    """Configuration for enhanced EP features."""
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
    **kwargs,
) -> EquiTileConfig:
    """Create a production-ready configuration.

    Sensible defaults for most use cases.
    """
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        mode="pc",  # PC mode for production
        dropout=0.1,
        weight_decay=1e-4,
        gradient_clip=1.0,
        **kwargs,
    )


def create_research_config(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    **kwargs,
) -> EquiTileConfig:
    """Create a research configuration for EP studies.

    Enables EP mode with enhanced features.
    """
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        mode="ep",  # EP mode for research
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
    **kwargs,
) -> EquiTileConfig:
    """Create a fast configuration for prototyping.

    Smaller model for quick iteration.
    """
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
    **kwargs,
) -> EnhancedEPConfig:
    """Create enhanced EP configuration."""
    return EnhancedEPConfig(
        use_layer_norm=use_layer_norm,
        use_curriculum=use_curriculum,
        curriculum_stages=curriculum_stages,
        **kwargs,
    )


def create_dynamic_config(
    growth_enabled: bool = True,
    prune_enabled: bool = True,
    **kwargs,
) -> DynamicEquiTileConfig:
    """Create dynamic tile configuration."""
    return DynamicEquiTileConfig(
        growth=TileGrowthConfig(
            growth_enabled=growth_enabled,
            prune_enabled=prune_enabled,
            **kwargs
        ),
        **kwargs,
    )
