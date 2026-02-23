"""
EquiTile Enhanced: Improved Scalable Local-Learning Architecture
================================================================

Enhanced version of EquiTile with configurable improvements for general-purpose ML:
- Layer normalization within tiles
- Residual error connections
- Per-tile adaptive learning rates
- Momentum for weight updates
- Deep network initialization
- Skip connections between non-adjacent layers
- Enhanced tile importance learning
- Batch normalization option
- Curriculum learning

All improvements are optional/configurable for ablation studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.models.base import BioModel, ModelConfig, register_model
from .config import EquiTileConfig, CurriculumConfig
from .core import TileState, EdgeParams, TileGraph

if TYPE_CHECKING:
    from torch import Tensor
    from .core import EquiTile


@dataclass
class EnhancedEquiTileConfig:
    """
    Enhanced configuration for EquiTile with all improvements.

    Standalone config with all options for ablation studies.
    """
    # =========================================================================
    # Core Architecture (from EquiTileConfig)
    # =========================================================================
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 4

    # Learning
    learning_rate: float = 0.01
    importance_lr: float = 0.001

    # Inference dynamics
    inference_steps: int = 10
    step_size: float = 0.1
    lambda_error: float = 0.1

    # EP mode parameters
    beta: float = 0.1
    beta_anneal: float = 1.0
    inference_steps_free: Optional[int] = None
    inference_steps_nudged: Optional[int] = None

    # Adaptive computation
    sparsity_threshold: float = 0.01
    min_active_fraction: float = 0.1

    # Regularization
    importance_decay: float = 0.95
    weight_decay: float = 1e-4
    dropout: float = 0.1
    gradient_clip: float = 1.0

    # Mode
    mode: str = "pc"

    # EP improvements
    use_symmetric_weights: bool = False
    clamp_activities: bool = True
    relaxation_tolerance: float = 1e-4

    # =========================================================================
    # Normalization Options (Enhanced)
    # =========================================================================
    use_layer_norm: bool = True
    """Apply layer normalization within each tile for stable signal propagation."""

    use_batch_norm: bool = False
    """Apply batch normalization across tile outputs (good for vision tasks)."""

    norm_eps: float = 1e-6
    """Epsilon for numerical stability in normalization."""

    # =========================================================================
    # Error Propagation Improvements
    # =========================================================================
    use_residual_errors: bool = True
    """Add residual error connections to prevent vanishing error signals."""

    residual_error_weight: float = 0.1
    """Weight for residual error flow from output to hidden tiles."""

    use_error_momentum: bool = False
    """Add momentum to error signals for smoother learning."""

    error_momentum: float = 0.9
    """Momentum coefficient for error signals."""

    # =========================================================================
    # Learning Rate Adaptation
    # =========================================================================
    per_tile_lr: bool = True
    """Enable per-tile adaptive learning rates."""

    lr_adaptation_rate: float = 0.01
    """Rate at which per-tile learning rates adapt."""

    lr_adaptation_decay: float = 0.99
    """Decay for per-tile learning rate running statistics."""

    min_lr_ratio: float = 0.1
    """Minimum learning rate as ratio of base LR."""

    max_lr_ratio: float = 10.0
    """Maximum learning rate as ratio of base LR."""

    # =========================================================================
    # Momentum for Weight Updates
    # =========================================================================
    use_weight_momentum: bool = True
    """Add momentum to Hebbian weight updates."""

    weight_momentum: float = 0.9
    """Momentum coefficient for weight updates."""

    # =========================================================================
    # Weight Initialization
    # =========================================================================
    deep_init: bool = True
    """Use deep-network-aware weight initialization."""

    init_scale_factor: float = 1.0
    """Additional scaling factor for weight initialization."""

    # =========================================================================
    # Architecture Improvements
    # =========================================================================
    use_skip_connections: bool = True
    """Add skip connections between non-adjacent layers (every 2 layers)."""

    skip_connection_weight: float = 0.5
    """Initial weight for skip connections."""

    # =========================================================================
    # Enhanced Tile Importance
    # =========================================================================
    enhanced_importance: bool = True
    """Use multi-factor importance learning (error + variance + sparsity)."""

    importance_competition: bool = True
    """Enable competitive importance (softmax across tiles)."""

    importance_entropy_weight: float = 0.01
    """Weight for entropy regularization in importance learning."""

    # =========================================================================
    # Activity Improvements
    # =========================================================================
    use_activity_clipping: bool = True
    """Clip activities to prevent explosion."""

    activity_clip_value: float = 5.0
    """Maximum absolute value for activity clipping."""

    use_activity_scaling: bool = False
    """Scale activities based on layer depth."""

    # =========================================================================
    # Gradient Improvements
    # =========================================================================
    use_gradient_centralization: bool = False
    """Centralize gradients for better convergence (good for vision)."""

    # =========================================================================
    # Curriculum Learning
    # =========================================================================
    use_curriculum: bool = False
    """Enable curriculum learning."""

    curriculum_stages: int = 5
    """Number of curriculum stages."""

    # =========================================================================
    # Monitoring and Debugging
    # =========================================================================
    track_tile_statistics: bool = True
    """Track per-tile statistics for analysis."""

    @classmethod
    def preset_minimal(cls) -> 'EnhancedEquiTileConfig':
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
    def preset_vision(cls) -> 'EnhancedEquiTileConfig':
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
    def preset_language(cls) -> 'EnhancedEquiTileConfig':
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
    def preset_rl(cls) -> 'EnhancedEquiTileConfig':
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


class CurriculumScheduler:
    """Curriculum learning scheduler for EP.

    Starts with easy examples and gradually increases difficulty.
    This helps EP converge better by providing cleaner learning signals early.
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.current_stage = 0
        self.samples_seen = 0
        self.stage_losses: List[float] = []
        self._difficulty_cache: Dict[int, float] = {}

    def get_sample_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model: 'EquiTile',
    ) -> torch.Tensor:
        if not self.config.enabled:
            return torch.ones(len(X))

        n_samples = len(X)

        # Estimate difficulty from prediction error
        # Note: This might be expensive, so we cache or use heuristics
        # For now, return ones if not using active selection
        return torch.ones(n_samples)

    def step(self, loss: float):
        """Update curriculum based on loss."""
        self.stage_losses.append(loss)
        self.samples_seen += 1

        if not self.config.auto_progress:
            return

        # Check if we should progress to next stage
        if self.samples_seen >= self.config.samples_per_stage:
            if len(self.stage_losses) >= 10:
                recent_improvement = (
                    sum(self.stage_losses[-10:-5]) - sum(self.stage_losses[-5:])
                ) / 5

                if recent_improvement < self.config.progress_threshold:
                    self.progress_stage()

            self.samples_seen = 0

    def progress_stage(self):
        """Progress to next curriculum stage."""
        if self.current_stage < self.config.n_stages - 1:
            self.current_stage += 1

    def reset(self):
        """Reset curriculum."""
        self.current_stage = 0
        self.samples_seen = 0
        self.stage_losses = []
        self._difficulty_cache = {}


class EnhancedTileGraph(TileGraph):
    """
    Enhanced tile graph with skip connections and improved initialization.
    """

    def build_layered(
        self,
        input_dim: int,
        output_dim: int,
        neurons_per_tile: int,
        num_hidden_layers: int,
        tiles_per_layer: int = 1,
        use_skip_connections: bool = True,
    ) -> None:
        """Build layered architecture with optional skip connections."""
        # Build base layered structure
        super().build_layered(
            input_dim, output_dim, neurons_per_tile, num_hidden_layers, tiles_per_layer
        )

        # Add skip connections (every 2 layers)
        if use_skip_connections and len(self.layer_ids) > 2:
            for layer_idx in range(len(self.layer_ids) - 2):
                for src_id in self.layer_ids[layer_idx]:
                    for dst_id in self.layer_ids[layer_idx + 2]:
                        # Only add if not already connected
                        if (src_id, dst_id) not in self.edges:
                            self._add_edge(src_id, dst_id)


class TileLayerNorm(nn.Module):
    """Layer normalization for individual tiles."""

    def __init__(self, neurons: int, eps: float = 1e-6):
        super().__init__()
        self.neurons = neurons
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(neurons))
        self.bias = nn.Parameter(torch.zeros(neurons))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, neurons)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps) * self.weight + self.bias


class BatchNormTile(nn.Module):
    """Batch normalization for tile outputs."""

    def __init__(self, dim: int, eps: float = 1e-6, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps, momentum=momentum)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x)


class EnhancedEdgeParams(EdgeParams):
    """Edge parameters with momentum buffer."""

    def __init__(
        self,
        src_id: int,
        dst_id: int,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_momentum: bool = True,
    ):
        super().__init__(src_id, dst_id, weight, bias)
        self.use_momentum = use_momentum
        if use_momentum:
            self.velocity_w = torch.zeros_like(weight) if weight is not None else None
            self.velocity_b = torch.zeros_like(bias) if bias is not None else None


@register_model("enhanced_equitile")
class EnhancedEquiTile(BioModel):
    """
    Enhanced EquiTile with configurable improvements.

    All improvements are optional via configuration for ablation studies.
    """

    algorithm_name = "EnhancedEquiTile"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        # Core architecture (required)
        neurons_per_tile: int,
        num_layers: int,
        tiles_per_layer: int,
        input_dim: int,
        output_dim: int,
        # Enhanced config
        enhanced_config: Optional[EnhancedEquiTileConfig] = None,
        # Backward compatibility with EquiTileConfig
        learning_rate: float = 0.01,
        importance_lr: float = 0.001,
        inference_steps: int = 10,
        step_size: float = 0.1,
        lambda_error: float = 0.1,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        gradient_clip: float = 1.0,
        activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu",
        task_type: Literal["classification", "regression", "binary", "multilabel"] = "classification",
        mode: Literal["pc", "ep"] = "pc",
        **kwargs,
    ):
        # Create base config with input/output dims for parent class
        if config is None:
            base_config = ModelConfig(
                name="enhanced_equitile",
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[neurons_per_tile * tiles_per_layer] * (max(0, num_layers - 2)),
                learning_rate=learning_rate,
            )
        else:
            base_config = config

        super().__init__(base_config, **kwargs)

        # Use enhanced config or create from parameters
        if enhanced_config is None:
            enhanced_config = EnhancedEquiTileConfig(
                neurons_per_tile=neurons_per_tile,
                num_layers=num_layers,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                inference_steps=inference_steps,
                step_size=step_size,
                lambda_error=lambda_error,
                weight_decay=weight_decay,
                dropout=dropout,
                gradient_clip=gradient_clip,
                mode=mode,
            )

        # Store enhanced config separately
        self.enhanced_config = enhanced_config
        self.config = enhanced_config  # Override for enhanced-specific access
        self.task_type = task_type
        self.mode = mode

        # Activation function
        self.activation = self._get_activation(activation)

        # Build enhanced graph
        self.graph = EnhancedTileGraph()
        num_hidden = max(0, num_layers - 2)
        self.graph.build_layered(
            input_dim, output_dim,
            neurons_per_tile, num_hidden, tiles_per_layer,
            use_skip_connections=self.config.use_skip_connections,
        )

        # I/O projections
        input_tile_dim = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.input_tile_ids
        )
        output_tile_dim = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.output_tile_ids
        )

        self.W_in = nn.Linear(input_dim, input_tile_dim)
        self.W_out = nn.Linear(output_tile_dim, output_dim)

        # Normalization layers
        self._build_normalization(input_tile_dim, output_tile_dim)

        # Tile importance (enhanced)
        self.tile_importance = nn.Parameter(torch.ones(len(self.graph.tiles)))
        self.edge_importance = nn.Parameter(torch.ones(len(self.graph.edges)))

        # Per-tile learning rate scales
        if self.config.per_tile_lr:
            self.tile_lr_scale = nn.Parameter(torch.zeros(len(self.graph.tiles)))
            self._tile_lr_running_mean = torch.zeros(len(self.graph.tiles))
            self._tile_lr_running_var = torch.ones(len(self.graph.tiles))

        # Per-edge momentum buffers
        if self.config.use_weight_momentum:
            for edge in self.graph.edges.values():
                if isinstance(edge, EnhancedEdgeParams):
                    pass  # Already has velocity buffers
                else:
                    # Upgrade to enhanced edge
                    enhanced_edge = EnhancedEdgeParams(
                        edge.src_id, edge.dst_id,
                        edge.weight, edge.bias,
                        use_momentum=True,
                    )
                    self.graph.edges[(edge.src_id, edge.dst_id)] = enhanced_edge

        # Error momentum buffers
        if self.config.use_error_momentum:
            self._error_momentum_buffer: Dict[int, Tensor] = {}

        # Curriculum Scheduler
        self.curriculum = None
        if self.config.use_curriculum:
            self.curriculum = CurriculumScheduler(
                CurriculumConfig(
                    enabled=True,
                    n_stages=self.config.curriculum_stages,
                )
            )

        # Tile statistics (for monitoring)
        if self.config.track_tile_statistics:
            self._tile_stats: Dict[int, Dict[str, float]] = {
                tile.id: {"activity_mean": 0, "error_mean": 0, "importance": 1}
                for tile in self.graph.all_tiles
            }

        # Optimizers
        self._optim_io = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=learning_rate,
        )
        self._optim_importance = torch.optim.Adam(
            [self.tile_importance, self.edge_importance] +
            ([self.tile_lr_scale] if self.config.per_tile_lr else []),
            lr=importance_lr,
        )

        # Dropout
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str):
        if name == "tanh":
            return torch.tanh
        elif name == "relu":
            return F.relu
        elif name == "silu":
            return F.silu
        return F.gelu

    def _build_normalization(self, input_dim: int, output_dim: int):
        """Build normalization layers."""
        self.layer_norms = nn.ModuleDict()
        self.batch_norms = nn.ModuleDict()

        if self.config.use_layer_norm:
            for tile in self.graph.all_tiles:
                if not tile.is_input:
                    self.layer_norms[str(tile.id)] = TileLayerNorm(
                        tile.neurons, eps=self.config.norm_eps
                    )

        if self.config.use_batch_norm:
            # Batch norm across concatenated tile outputs
            hidden_dim = sum(
                self.graph.tiles[tid].neurons
                for tid in self.graph.all_tiles
                if not tid in self.graph.input_tile_ids
            )
            if hidden_dim > 0:
                self.batch_norms["hidden"] = BatchNormTile(
                    hidden_dim, eps=self.config.norm_eps
                )

    def _init_weights(self) -> None:
        """Initialize weights with deep-network-aware initialization."""
        device = next(self.parameters()).device
        num_layers = len(self.graph.layer_ids)

        with torch.no_grad():
            for edge_idx, edge in enumerate(self.graph.edges.values()):
                if edge.weight is not None:
                    fan_in, fan_out = edge.weight.shape

                    if self.config.deep_init:
                        # Deep network initialization
                        depth_scale = math.sqrt(2.0 / (fan_in + fan_out))
                        layer_factor = math.sqrt(2.0 / max(1, num_layers - 1))
                        std = depth_scale * layer_factor * self.config.init_scale_factor
                    else:
                        # Standard fan-in initialization
                        std = math.sqrt(2.0 / fan_in) * self.config.init_scale_factor

                    edge.weight.normal_(0, std)

                if edge.bias is not None:
                    nn.init.zeros_(edge.bias)

            # I/O projections
            nn.init.kaiming_normal_(self.W_in.weight, mode='fan_in', nonlinearity='relu')
            if self.W_in.bias is not None:
                nn.init.zeros_(self.W_in.bias)

            if self.config.deep_init:
                # Scale output projection for deep networks
                output_scale = math.sqrt(2.0 / num_layers)
                nn.init.xavier_normal_(self.W_out.weight, gain=output_scale)
            else:
                nn.init.xavier_normal_(self.W_out.weight, gain=1.0)

            if self.W_out.bias is not None:
                nn.init.zeros_(self.W_out.bias)

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        device = next(self.parameters()).device

        with torch.no_grad():
            for edge in self.graph.edges.values():
                if edge.weight is not None:
                    edge.weight = edge.weight.to(device)
                if edge.bias is not None:
                    edge.bias = edge.bias.to(device)
                if isinstance(edge, EnhancedEdgeParams):
                    if edge.velocity_w is not None:
                        edge.velocity_w = edge.velocity_w.to(device)
                    if edge.velocity_b is not None:
                        edge.velocity_b = edge.velocity_b.to(device)

        # Move momentum buffers
        if self.config.use_error_momentum:
            for key in self._error_momentum_buffer:
                self._error_momentum_buffer[key] = self._error_momentum_buffer[key].to(device)

        return model

    def _apply_activation(self, x: Tensor) -> Tensor:
        return self._dropout(self.activation(x))

    def _normalize_tile_activity(self, tile: TileState, batch_size: int, device: torch.device):
        """Apply normalization to tile activity."""
        if tile.activity is None or tile.is_input:
            return

        # Layer normalization
        if self.config.use_layer_norm and str(tile.id) in self.layer_norms:
            tile.activity = self.layer_norms[str(tile.id)](tile.activity)

        # Activity clipping
        if self.config.use_activity_clipping:
            tile.activity = torch.clamp(
                tile.activity,
                -self.config.activity_clip_value,
                self.config.activity_clip_value,
            )

        # Activity scaling based on depth
        if self.config.use_activity_scaling:
            depth_scale = math.sqrt(2.0 / (tile.layer_id + 1))
            tile.activity = tile.activity * depth_scale

    def _compute_predictions(self, batch_size: int, device: torch.device) -> None:
        """Compute predictions with normalization."""
        for tile in self.graph.all_tiles:
            if tile.is_input:
                continue

            pred = torch.zeros(batch_size, tile.neurons, device=device)

            for src_id in tile.bwd_neighbors:
                edge = self.graph.edges.get((src_id, tile.id))
                if edge is None or edge.weight is None:
                    continue

                src = self.graph.tiles[src_id]
                src_activity = src.activity if src.activity is not None else torch.zeros(
                    batch_size, src.neurons, device=device
                )
                pred = pred + self._apply_activation(src_activity) @ edge.weight

            if edge and edge.bias is not None:
                pred = pred + edge.bias.unsqueeze(0)

            tile.prediction = pred

    def _compute_errors(self) -> None:
        """Compute errors with momentum option."""
        for tile in self.graph.all_tiles:
            if tile.activity is None:
                continue

            if tile.prediction is None:
                tile.error = tile.activity.clone()
            else:
                tile.error = tile.activity - tile.prediction

            # Apply error momentum
            if self.config.use_error_momentum:
                if tile.id not in self._error_momentum_buffer:
                    self._error_momentum_buffer[tile.id] = torch.zeros_like(tile.error)

                self._error_momentum_buffer[tile.id] = (
                    self.config.error_momentum * self._error_momentum_buffer[tile.id] +
                    (1 - self.config.error_momentum) * tile.error
                )
                tile.error = self._error_momentum_buffer[tile.id]

            # Track statistics
            if self.config.track_tile_statistics:
                err_norm = tile.error.norm(p=2, dim=-1).mean().item()
                self._tile_stats[tile.id]["error_mean"] = err_norm

    def _propagate_errors(self, tile_errors: Dict[int, Tensor]) -> None:
        """Propagate errors backward with residual connections."""
        hidden_tiles = sorted(
            [t for t in self.graph.all_tiles if not t.is_output and not t.is_input],
            key=lambda t: -t.layer_id,
        )

        for tile in hidden_tiles:
            error = torch.zeros_like(tile.activity)

            # Standard error from forward neighbors
            for fwd_id in tile.fwd_neighbors:
                if fwd_id not in tile_errors:
                    continue
                edge = self.graph.edges.get((tile.id, fwd_id))
                if edge and edge.weight is not None:
                    error = error + tile_errors[fwd_id] @ edge.weight.T

            # Residual error connections (direct from output)
            # Only add if dimensions match (same neurons per tile)
            if self.config.use_residual_errors:
                for out_tile_id in self.graph.output_tile_ids:
                    if out_tile_id not in tile_errors:
                        continue
                    out_error = tile_errors[out_tile_id]

                    # Check if dimensions are compatible
                    if out_error.shape[-1] == error.shape[-1]:
                        error = error + self.config.residual_error_weight * out_error
                    elif self.config.use_skip_connections:
                        # Skip connections already provide residual flow,
                        # so residual errors are handled through the graph
                        pass

            tile_errors[tile.id] = error

    def _get_tile_learning_rate(self, tile_idx: int, tile: TileState) -> float:
        """Get per-tile adaptive learning rate."""
        if not self.config.per_tile_lr:
            return self.config.learning_rate

        # Get base scale
        scale = torch.sigmoid(self.tile_lr_scale[tile_idx]).item()

        # Update running statistics for adaptation
        if self.config.track_tile_statistics:
            error_mean = self._tile_stats.get(tile.id, {}).get("error_mean", 0)

            # Update running mean
            self._tile_lr_running_mean[tile_idx] = (
                self.config.lr_adaptation_decay * self._tile_lr_running_mean[tile_idx] +
                (1 - self.config.lr_adaptation_decay) * error_mean
            )

        # Scale learning rate based on tile importance and error
        base_lr = self.config.learning_rate
        adapted_lr = base_lr * (0.5 + scale)

        # Clamp to valid range
        min_lr = base_lr * self.config.min_lr_ratio
        max_lr = base_lr * self.config.max_lr_ratio
        adapted_lr = max(min_lr, min(max_lr, adapted_lr))

        return adapted_lr

    def _update_importance(self) -> None:
        """Update tile and edge importance with enhanced learning."""
        self._optim_importance.zero_grad()

        if self.config.enhanced_importance:
            # Multi-factor importance learning
            tile_loss = torch.tensor(0.0, device=self.tile_importance.device)
            reg_loss = torch.tensor(0.0, device=self.tile_importance.device)

            for i, tile in enumerate(self.graph.all_tiles):
                if tile.error is None:
                    continue

                err_norm = tile.error.norm(p=2, dim=-1).mean()
                imp = torch.sigmoid(self.tile_importance[i])

                # Factor 1: Error-driven signal
                tile_loss = tile_loss + imp * err_norm.detach()

                # Factor 2: Gradient variance (high variance → important)
                if self.config.track_tile_statistics:
                    grad_var = tile.error.var()
                    tile_loss = tile_loss + imp * grad_var.detach() * 0.1

                # Factor 3: Sparsity regularization
                reg_loss = reg_loss + 0.01 * ((imp - 0.5) ** 2)

            # Entropy regularization for competition
            if self.config.importance_competition:
                importance_probs = F.softmax(self.tile_importance, dim=0)
                entropy = -(importance_probs * torch.log(importance_probs + 1e-8)).sum()
                reg_loss = reg_loss - self.config.importance_entropy_weight * entropy

            total_loss = tile_loss + reg_loss
            total_loss.backward()
        else:
            # Simple importance update (baseline)
            for i, tile in enumerate(self.graph.all_tiles):
                if tile.error is None:
                    continue

                err_norm = tile.error.norm(p=2, dim=-1).mean()
                imp = torch.sigmoid(self.tile_importance[i])
                tile_loss = imp * err_norm.detach()
                tile_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            [self.tile_importance, self.edge_importance] +
            ([self.tile_lr_scale] if self.config.per_tile_lr else []),
            max_norm=1.0,
        )

        self._optim_importance.step()

        # Update tracked importance values
        for i, tile in enumerate(self.graph.all_tiles):
            if self.config.track_tile_statistics:
                self._tile_stats[tile.id]["importance"] = torch.sigmoid(self.tile_importance[i]).item()

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with all enhancements."""
        if self.mode == "ep":
            return self._train_step_ep(x, y)
        return self._train_step_pc(x, y)

    def _train_step_pc(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with predictive coding and enhancements."""
        batch, device = x.shape[0], x.device

        input_proj = self.W_in(x)

        # Initialize tiles
        for tile in self.graph.all_tiles:
            if tile.is_input:
                idx = self.graph.input_tile_ids.index(tile.id)
                start = idx * self.config.neurons_per_tile
                tile.activity = input_proj[:, start:start + tile.neurons].clone()
            else:
                tile.activity = torch.zeros(batch, tile.neurons, device=device)
            tile.prediction = None
            tile.error = None

        # Inference phase
        for _ in range(self.config.inference_steps):
            self._compute_predictions(batch, device)
            self._compute_errors()

            # Update activities with normalization
            for i, tile in enumerate(self.graph.all_tiles):
                if tile.is_input:
                    idx = self.graph.input_tile_ids.index(tile.id)
                    start = idx * self.config.neurons_per_tile
                    tile.activity = input_proj[:, start:start + tile.neurons].clone()
                    continue

                if tile.error is None:
                    continue

                # Get per-tile learning rate
                lr = self._get_tile_learning_rate(i, tile)
                imp = torch.sigmoid(self.tile_importance[i]).item()

                grad = tile.error + self.config.lambda_error * tile.activity

                # Top-down modulation
                for dst_id in tile.fwd_neighbors:
                    dst = self.graph.tiles[dst_id]
                    edge = self.graph.edges.get((tile.id, dst_id))
                    if edge and edge.weight is not None and dst.error is not None:
                        grad = grad + dst.error @ edge.weight.T

                delta = self.config.step_size * imp * grad
                tile.activity = tile.activity - delta

                # Normalize activity
                self._normalize_tile_activity(tile, batch, device)

        # Task-driven learning
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1,
        )

        # Apply batch normalization
        if self.config.use_batch_norm and "hidden" in self.batch_norms:
            out_activities = self.batch_norms["hidden"](out_activities)

        logits = self.W_out(out_activities)

        # Compute loss
        if self.task_type == "regression":
            y_target = y.float()
            if y_target.dim() < logits.dim():
                y_target = y_target.unsqueeze(-1)
            loss = F.mse_loss(logits, y_target)
            output_delta = (logits - y_target) @ self.W_out.weight
        elif self.task_type == "binary":
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            output_delta = (logits.sigmoid() - y.float()) @ self.W_out.weight
        else:
            loss = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=-1)
            target_onehot = F.one_hot(y, self.output_dim).float().to(device)
            output_delta = (probs - target_onehot) @ self.W_out.weight

        # Update I/O projections
        self._optim_io.zero_grad()
        loss.backward()

        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.W_in.parameters()) + list(self.W_out.parameters()),
                self.config.gradient_clip,
            )
        self._optim_io.step()

        # Local Hebbian updates with momentum
        tile_errors: Dict[int, Tensor] = {}

        for i, tile_id in enumerate(self.graph.output_tile_ids):
            tile = self.graph.tiles[tile_id]
            start = i * self.config.neurons_per_tile
            end = start + tile.neurons
            tile_errors[tile_id] = output_delta[:, start:end].clone()

        # Propagate errors with residual connections
        self._propagate_errors(tile_errors)

        # Weight updates
        lr = self.config.learning_rate
        with torch.no_grad():
            for edge_idx, (edge_key, edge) in enumerate(self.graph.edges.items()):
                src_id, dst_id = edge_key
                src = self.graph.tiles[src_id]
                dst = self.graph.tiles[dst_id]

                if src.activity is None or dst.id not in tile_errors:
                    continue

                imp = torch.sigmoid(self.edge_importance[edge_idx]).item()
                src_act = self._apply_activation(src.activity)
                dst_err = tile_errors[dst.id]

                weight_update = imp * (src_act.T @ dst_err) / batch
                bias_update = imp * dst_err.mean(dim=0) / batch

                # Gradient centralization (for vision)
                if self.config.use_gradient_centralization:
                    weight_update = weight_update - weight_update.mean()

                # Apply momentum
                if self.config.use_weight_momentum and isinstance(edge, EnhancedEdgeParams):
                    edge.velocity_w = (
                        self.config.weight_momentum * edge.velocity_w + weight_update
                    )
                    edge.velocity_b = (
                        self.config.weight_momentum * edge.velocity_b + bias_update
                    )

                    if edge.weight is not None:
                        edge.weight.data = edge.weight.data - lr * (
                            edge.velocity_w + self.config.weight_decay * edge.weight.data
                        )
                    if edge.bias is not None:
                        edge.bias.data = edge.bias.data - lr * edge.velocity_b
                else:
                    # Standard update without momentum
                    if edge.weight is not None:
                        edge.weight.data = edge.weight.data - lr * (
                            weight_update + self.config.weight_decay * edge.weight.data
                        )
                    if edge.bias is not None:
                        edge.bias.data = edge.bias.data - lr * bias_update

        # Update importance
        self._update_importance()

        # Update curriculum
        if self.curriculum is not None:
            self.curriculum.step(loss.item())

        # Compute metrics
        with torch.no_grad():
            if self.task_type == "classification":
                accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
            elif self.task_type == "binary":
                accuracy = (logits.sigmoid() > 0.5).long().squeeze(-1)
                accuracy = (accuracy == y).float().mean().item()
            else:
                accuracy = 0.0

        active_tiles = sum(
            1 for t in self.graph.all_tiles
            if self._tile_stats.get(t.id, {}).get("error_mean", 0) > self.config.sparsity_threshold
        )

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "mean_error": sum(
                self._tile_stats.get(t.id, {}).get("error_mean", 0)
                for t in self.graph.all_tiles
            ) / len(self.graph.tiles),
            "active_tiles": active_tiles,
            "active_tiles_pct": active_tiles / len(self.graph.tiles) * 100,
            "mode": self.mode,
        }

    def _train_step_ep(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with equilibrium propagation (fallback to base implementation)."""
        # For now, use base EP implementation
        # Can be enhanced similarly to PC mode
        raise NotImplementedError("EP mode not yet enhanced. Use mode='pc'.")

    def forward(self, x: Tensor, steps: Optional[int] = None) -> Tensor:
        """Forward pass with normalization."""
        batch, device = x.shape[0], x.device
        steps = steps if steps is not None else self.config.inference_steps

        input_proj = self.W_in(x)

        for tile in self.graph.all_tiles:
            if tile.is_input:
                idx = self.graph.input_tile_ids.index(tile.id)
                start = idx * self.config.neurons_per_tile
                tile.activity = input_proj[:, start:start + tile.neurons].clone()
            else:
                tile.activity = torch.zeros(batch, tile.neurons, device=device)
            tile.prediction = None
            tile.error = None

        for _ in range(steps):
            self._compute_predictions(batch, device)
            self._compute_errors()

            for i, tile in enumerate(self.graph.all_tiles):
                if tile.is_input or tile.error is None:
                    continue
                imp = torch.sigmoid(self.tile_importance[i]).item()
                grad = tile.error + self.config.lambda_error * tile.activity

                for dst_id in tile.fwd_neighbors:
                    dst = self.graph.tiles[dst_id]
                    edge = self.graph.edges.get((tile.id, dst_id))
                    if edge and edge.weight is not None and dst.error is not None:
                        grad = grad + dst.error @ edge.weight.T

                delta = self.config.step_size * imp * grad
                tile.activity = tile.activity - delta
                self._normalize_tile_activity(tile, batch, device)

        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1,
        )

        if self.config.use_batch_norm and "hidden" in self.batch_norms:
            out_activities = self.batch_norms["hidden"](out_activities)

        return self.W_out(out_activities)

    def get_stats(self) -> Dict:
        """Get enhanced statistics including tile-level metrics."""
        stats = super().get_stats()

        if self.config.track_tile_statistics:
            stats["tile_statistics"] = self._tile_stats.copy()

            # Per-layer summaries
            layer_stats = {}
            for layer_idx, layer_tiles in enumerate(self.graph.layer_ids):
                layer_errors = [
                    self._tile_stats.get(tid, {}).get("error_mean", 0)
                    for tid in layer_tiles
                ]
                layer_importance = [
                    self._tile_stats.get(tid, {}).get("importance", 1)
                    for tid in layer_tiles
                ]
                layer_stats[f"layer_{layer_idx}"] = {
                    "mean_error": sum(layer_errors) / len(layer_errors),
                    "mean_importance": sum(layer_importance) / len(layer_importance),
                }
            stats["layer_statistics"] = layer_stats

        return stats

    def summarize(self) -> str:
        """Get model summary with enhancement status."""
        lines = [
            "=" * 60,
            "Enhanced EquiTile Model Summary",
            "=" * 60,
            f"Architecture: {len(self.graph.layer_ids)} layers, "
            f"{len(self.graph.tiles)} tiles, "
            f"{len(self.graph.edges)} edges",
            f"Parameters: {sum(p.numel() for p in self.parameters()):,}",
            "",
            "Enabled Enhancements:",
        ]

        enhancements = [
            ("Layer Normalization", self.config.use_layer_norm),
            ("Batch Normalization", self.config.use_batch_norm),
            ("Residual Errors", self.config.use_residual_errors),
            ("Error Momentum", self.config.use_error_momentum),
            ("Per-tile LR", self.config.per_tile_lr),
            ("Weight Momentum", self.config.use_weight_momentum),
            ("Deep Init", self.config.deep_init),
            ("Skip Connections", self.config.use_skip_connections),
            ("Enhanced Importance", self.config.enhanced_importance),
            ("Activity Clipping", self.config.use_activity_clipping),
            ("Gradient Centralization", self.config.use_gradient_centralization),
            ("Curriculum Learning", self.config.use_curriculum),
        ]

        for name, enabled in enhancements:
            status = "✓" if enabled else "✗"
            lines.append(f"  [{status}] {name}")

        return "\n".join(lines)


def create_enhanced_model(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    input_dim: int = 784,
    output_dim: int = 10,
    use_layer_norm: bool = True,
    use_curriculum: bool = True,
    **kwargs,
) -> EnhancedEquiTile:
    """Create an enhanced EquiTile model.

    Factory function for backward compatibility.
    """
    enhanced_config = EnhancedEquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        use_layer_norm=use_layer_norm,
        use_curriculum=use_curriculum,
        **kwargs
    )

    return EnhancedEquiTile(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        input_dim=input_dim,
        output_dim=output_dim,
        enhanced_config=enhanced_config,
        **kwargs
    )

EnhancedEPConfig = EnhancedEquiTileConfig
