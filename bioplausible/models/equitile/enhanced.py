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
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.models.base import ModelConfig, register_model

from .config import CurriculumConfig, EnhancedEquiTileConfig
from .core import EquiTile
from .kernels import (
    compute_activity_update,
    compute_hebbian_update,
)
from .topology import TileState
from .utils.init_utils import initialize_edge_weights, initialize_io_projections

if TYPE_CHECKING:
    from torch import Tensor


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
        model: "EquiTile",
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


@register_model("enhanced_equitile")
class EnhancedEquiTile(EquiTile):
    """
    Enhanced EquiTile with configurable improvements.

    All improvements are optional via configuration for ablation studies.
    Inherits from EquiTile.
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
        # Backward compatibility
        learning_rate: float = 0.01,
        importance_lr: float = 0.001,
        inference_steps: int = 10,
        step_size: float = 0.1,
        lambda_error: float = 0.1,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        gradient_clip: float = 1.0,
        activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu",
        task_type: Literal[
            "classification", "regression", "binary", "multilabel"
        ] = "classification",
        mode: Literal["pc", "ep"] = "pc",
        **kwargs,
    ):
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
                task_type=task_type,
                activation=activation,
                **kwargs,  # Pass remaining kwargs to config
            )

        # Store enhanced config as self.equitile_config (parent expects this)
        # Parent __init__ will set self.equitile_config = config

        super().__init__(
            config=enhanced_config, input_dim=input_dim, output_dim=output_dim, **kwargs
        )

        # Normalization layers
        input_tile_dim = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.input_tile_ids
        )
        output_tile_dim = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.output_tile_ids
        )
        self._build_normalization(input_tile_dim, output_tile_dim)

        # Momentum buffers
        self.edge_velocity_w: Dict[str, Tensor] = {}
        self.edge_velocity_b: Dict[str, Tensor] = {}
        self._init_momentum_buffers()

        # Per-tile learning rate scales
        if self.equitile_config.per_tile_lr:
            self.tile_lr_scale = nn.Parameter(torch.zeros(len(self.graph.tiles)))
            self._tile_lr_running_mean = torch.zeros(len(self.graph.tiles))
            self._tile_lr_running_var = torch.ones(len(self.graph.tiles))

            # Re-setup optimizers to include new parameter
            self.reset_optimizers()

        # Error momentum buffers
        if self.equitile_config.use_error_momentum:
            self._error_momentum_buffer: Dict[int, Tensor] = {}

        # Curriculum Scheduler
        self.curriculum = None
        if self.equitile_config.use_curriculum:
            self.curriculum = CurriculumScheduler(
                CurriculumConfig(
                    enabled=True,
                    n_stages=self.equitile_config.curriculum_stages,
                )
            )

        # Tile statistics (for monitoring)
        if self.equitile_config.track_tile_statistics:
            self._tile_stats: Dict[int, Dict[str, float]] = {
                tile.id: {"activity_mean": 0, "error_mean": 0, "importance": 1}
                for tile in self.graph.all_tiles
            }

    def _build_normalization(self, input_dim: int, output_dim: int):
        """Build normalization layers."""
        self.layer_norms = nn.ModuleDict()
        self.batch_norms = nn.ModuleDict()

        if self.equitile_config.use_layer_norm:
            for tile in self.graph.all_tiles:
                if not tile.is_input:
                    self.layer_norms[str(tile.id)] = TileLayerNorm(
                        tile.neurons, eps=self.equitile_config.norm_eps
                    )

        if self.equitile_config.use_batch_norm:
            # Batch norm across concatenated tile outputs
            hidden_dim = sum(
                self.graph.tiles[tid].neurons for tid in self.graph.output_tile_ids
            )
            if hidden_dim > 0:
                self.batch_norms["hidden"] = BatchNormTile(
                    hidden_dim, eps=self.equitile_config.norm_eps
                )

    def _init_momentum_buffers(self):
        """Initialize momentum buffers."""
        if self.equitile_config.use_weight_momentum:
            for src, dst in self.graph.edges:
                key = f"edge_{src}_{dst}"
                weight = self.edge_weights[key]
                bias = self.edge_biases[key]

                self.edge_velocity_w[key] = torch.zeros_like(weight)
                self.edge_velocity_b[key] = torch.zeros_like(bias)

    def _reset_weights(self) -> None:
        """Reset weights with deep-network-aware initialization (Overrides base)."""
        num_layers = len(self.graph.layer_ids)

        with torch.no_grad():
            for key, weight in self.edge_weights.items():
                initialize_edge_weights(
                    weight,
                    bias=None,
                    init_type="normal",
                    gain=self.equitile_config.init_scale_factor,
                    deep_init=self.equitile_config.deep_init,
                    num_layers=num_layers,
                )

            for key, bias in self.edge_biases.items():
                nn.init.zeros_(bias)

            initialize_io_projections(
                self.W_in,
                self.W_out,
                deep_init=self.equitile_config.deep_init,
                num_layers=num_layers,
            )

    def _setup_optimizers(self) -> None:
        """Initialize optimizers (Overrides base to include tile_lr_scale)."""
        # I/O Optimizer
        self._optim_io = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=self.equitile_config.learning_rate,
        )

        # Importance Optimizer
        params = [self.tile_importance, self.edge_importance]
        if hasattr(self, "tile_lr_scale"):
            params.append(self.tile_lr_scale)

        self._optim_importance = torch.optim.Adam(
            params,
            lr=self.equitile_config.importance_lr,
        )

        # Full Optimizer
        if self.equitile_config.mode in ("backprop", "ep"):
            self._optim_full = torch.optim.Adam(
                self.parameters(), lr=self.equitile_config.learning_rate
            )

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        device = next(self.parameters()).device

        # Move momentum buffers
        if self.config.use_weight_momentum:
            for key in self.edge_velocity_w:
                self.edge_velocity_w[key] = self.edge_velocity_w[key].to(device)
            for key in self.edge_velocity_b:
                self.edge_velocity_b[key] = self.edge_velocity_b[key].to(device)

        if self.config.use_error_momentum:
            for key in self._error_momentum_buffer:
                self._error_momentum_buffer[key] = self._error_momentum_buffer[key].to(
                    device
                )

        # Move running stats
        if hasattr(self, "_tile_lr_running_mean"):
            self._tile_lr_running_mean = self._tile_lr_running_mean.to(device)
            self._tile_lr_running_var = self._tile_lr_running_var.to(device)

        return model

    def _normalize_tile_activity(self, tile: TileState):
        """Apply normalization to tile activity."""
        if tile.activity is None or tile.is_input:
            return

        # Layer normalization
        if self.equitile_config.use_layer_norm and str(tile.id) in self.layer_norms:
            tile.activity = self.layer_norms[str(tile.id)](tile.activity)

        # Activity clipping
        if self.equitile_config.use_activity_clipping:
            tile.activity = torch.clamp(
                tile.activity,
                -self.equitile_config.activity_clip_value,
                self.equitile_config.activity_clip_value,
            )

        # Activity scaling based on depth
        if self.equitile_config.use_activity_scaling:
            depth_scale = math.sqrt(2.0 / (tile.layer_id + 1))
            tile.activity = tile.activity * depth_scale

    def _update_tile_activity(self, tile: TileState, delta: Tensor, clamp: bool):
        """Update tile activity with delta and normalization."""
        tile.activity = tile.activity - delta
        # Use our enhanced normalization (which handles clipping)
        self._normalize_tile_activity(tile)

    def _compute_errors(self) -> None:
        """Compute errors with momentum option."""
        # Base logic for error computation
        super()._compute_errors()

        # Apply momentum
        if self.equitile_config.use_error_momentum:
            for tile in self.graph.all_tiles:
                if tile.error is None:
                    continue

                if tile.id not in self._error_momentum_buffer:
                    self._error_momentum_buffer[tile.id] = torch.zeros_like(tile.error)

                self._error_momentum_buffer[tile.id] = (
                    self.equitile_config.error_momentum
                    * self._error_momentum_buffer[tile.id]
                    + (1 - self.equitile_config.error_momentum) * tile.error
                )
                tile.error = self._error_momentum_buffer[tile.id]

        # Track statistics
        if self.equitile_config.track_tile_statistics:
            for tile in self.graph.all_tiles:
                if tile.error is not None:
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
                weight, _ = self._get_edge_params(tile.id, fwd_id)
                if weight is not None:
                    error = error + tile_errors[fwd_id] @ weight.T

            # Residual error connections (direct from output)
            if self.equitile_config.use_residual_errors:
                for out_tile_id in self.graph.output_tile_ids:
                    if out_tile_id not in tile_errors:
                        continue
                    out_error = tile_errors[out_tile_id]

                    if out_error.shape[-1] == error.shape[-1]:
                        error = (
                            error
                            + self.equitile_config.residual_error_weight * out_error
                        )

            tile_errors[tile.id] = error

    def _get_tile_learning_rate(self, tile_idx: int, tile: TileState) -> float:
        """Get per-tile adaptive learning rate."""
        if not self.equitile_config.per_tile_lr:
            return self.equitile_config.learning_rate

        # Get base scale
        scale = torch.sigmoid(self.tile_lr_scale[tile_idx]).item()

        # Update running statistics for adaptation
        if self.equitile_config.track_tile_statistics:
            error_mean = self._tile_stats.get(tile.id, {}).get("error_mean", 0)

            # Update running mean
            self._tile_lr_running_mean[tile_idx] = (
                self.equitile_config.lr_adaptation_decay
                * self._tile_lr_running_mean[tile_idx]
                + (1 - self.equitile_config.lr_adaptation_decay) * error_mean
            )

        # Scale learning rate based on tile importance and error
        base_lr = self.equitile_config.learning_rate
        adapted_lr = base_lr * (0.5 + scale)

        # Clamp to valid range
        min_lr = base_lr * self.equitile_config.min_lr_ratio
        max_lr = base_lr * self.equitile_config.max_lr_ratio
        adapted_lr = max(min_lr, min(max_lr, adapted_lr))

        return adapted_lr

    def _update_importance(self) -> None:
        """Update tile and edge importance with enhanced learning."""
        if not self.equitile_config.enhanced_importance:
            super()._update_importance()
            return

        self._optim_importance.zero_grad()

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
            if self.equitile_config.track_tile_statistics:
                grad_var = tile.error.var()
                tile_loss = tile_loss + imp * grad_var.detach() * 0.1

            # Factor 3: Sparsity regularization
            reg_loss = reg_loss + 0.01 * ((imp - 0.5) ** 2)

        # Entropy regularization for competition
        if self.equitile_config.importance_competition:
            importance_probs = F.softmax(self.tile_importance, dim=0)
            entropy = -(importance_probs * torch.log(importance_probs + 1e-8)).sum()
            reg_loss = (
                reg_loss - self.equitile_config.importance_entropy_weight * entropy
            )

        total_loss = tile_loss + reg_loss
        total_loss.backward()

        # Clip gradients
        params = [self.tile_importance, self.edge_importance]
        if hasattr(self, "tile_lr_scale"):
            params.append(self.tile_lr_scale)

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        self._optim_importance.step()

        # Update tracked importance values
        for i, tile in enumerate(self.graph.all_tiles):
            if self.equitile_config.track_tile_statistics:
                self._tile_stats[tile.id]["importance"] = torch.sigmoid(
                    self.tile_importance[i]
                ).item()

    def _relaxation_step(
        self, step_size: float, clamp: bool, output_nudge: Optional[Tensor] = None
    ):
        """Perform a single relaxation step with per-tile LR."""
        for i, tile in enumerate(self.graph.all_tiles):
            if tile.is_input or tile.error is None:
                continue

            # Enhanced: per-tile LR
            lr_scale = 1.0
            if self.equitile_config.per_tile_lr:
                lr = self._get_tile_learning_rate(i, tile)
                lr_scale = lr / self.equitile_config.learning_rate

            imp = torch.sigmoid(self.tile_importance[i]).item()

            fwd_feedback = []
            for dst_id in tile.fwd_neighbors:
                dst = self.graph.tiles[dst_id]
                weight, _ = self._get_edge_params(tile.id, dst_id)
                if weight is not None and dst.error is not None:
                    fwd_feedback.append(dst.error @ weight.T)

            # We use clamp=False because _update_tile_activity handles normalization and clamping
            new_activity = compute_activity_update(
                activity=tile.activity,
                error=tile.error,
                fwd_feedback=fwd_feedback,
                importance=imp,
                step_size=step_size * lr_scale,
                lambda_error=self.equitile_config.lambda_error,
                clamp_min=0.0,  # Unused when clamp=False
                clamp_max=0.0,  # Unused when clamp=False
                clamp=False,
            )

            # compute_activity_update returns `activity - delta`
            # _update_tile_activity expects `delta`
            # So delta = activity - new_activity
            delta = tile.activity - new_activity

            self._update_tile_activity(tile, delta, clamp)

        if output_nudge is not None:
            for i, tile_id in enumerate(self.graph.output_tile_ids):
                tile = self.graph.tiles[tile_id]
                if tile.activity is not None:
                    start = i * self.equitile_config.neurons_per_tile
                    end = start + tile.neurons
                    if end <= output_nudge.shape[1]:
                        # Nudge is additive
                        delta = -self.equitile_config.beta * output_nudge[:, start:end]
                        self._update_tile_activity(tile, delta, clamp)

    def _pc_learning(self, x: Tensor, y: Tensor, batch: int) -> Dict[str, float]:
        """Run PC learning phase with enhancements."""
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1,
        )

        # Enhanced: Batch Norm
        if self.equitile_config.use_batch_norm and "hidden" in self.batch_norms:
            out_activities = self.batch_norms["hidden"](out_activities)

        logits = self.W_out(out_activities)
        loss, output_delta = self._compute_loss_and_delta(logits, y)

        # Update I/O
        self._optim_io.zero_grad()
        loss.backward()
        if self.equitile_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.W_in.parameters()) + list(self.W_out.parameters()),
                self.equitile_config.gradient_clip,
            )
        self._optim_io.step()

        # Local Updates with Momentum
        tile_errors: Dict[int, Tensor] = {}
        for i, tile_id in enumerate(self.graph.output_tile_ids):
            tile = self.graph.tiles[tile_id]
            start = i * self.equitile_config.neurons_per_tile
            end = start + tile.neurons
            tile_errors[tile_id] = output_delta[:, start:end].clone()

        # Enhanced: Propagate errors (residual)
        self._propagate_errors(tile_errors)

        lr = self.equitile_config.learning_rate
        with torch.no_grad():
            for edge_idx, (src_id, dst_id) in enumerate(self.graph.edges):
                weight, bias = self._get_edge_params(src_id, dst_id)
                key = f"edge_{src_id}_{dst_id}"

                src = self.graph.tiles[src_id]
                dst = self.graph.tiles[dst_id]

                if src.activity is None or dst.id not in tile_errors:
                    continue

                imp = torch.sigmoid(self.edge_importance[edge_idx]).item()
                src_act = self._apply_activation(src.activity)
                dst_err = tile_errors[dst.id]

                weight_update, bias_update = compute_hebbian_update(
                    src_act, dst_err, imp, batch
                )

                # Enhanced: Gradient centralization
                if self.equitile_config.use_gradient_centralization:
                    weight_update = weight_update - weight_update.mean()

                # Enhanced: Momentum
                if self.equitile_config.use_weight_momentum:
                    self.edge_velocity_w[key] = (
                        self.equitile_config.weight_momentum * self.edge_velocity_w[key]
                        + weight_update
                    )
                    self.edge_velocity_b[key] = (
                        self.equitile_config.weight_momentum * self.edge_velocity_b[key]
                        + bias_update
                    )

                    if weight is not None:
                        weight.data = weight.data - lr * (
                            self.edge_velocity_w[key]
                            + self.equitile_config.weight_decay * weight.data
                        )
                    if bias is not None:
                        bias.data = bias.data - lr * self.edge_velocity_b[key]
                else:
                    if weight is not None:
                        weight.data = weight.data - lr * (
                            weight_update
                            + self.equitile_config.weight_decay * weight.data
                        )
                    if bias is not None:
                        bias.data = bias.data - lr * bias_update

        self._update_importance()

        if self.curriculum is not None:
            self.curriculum.step(loss.item())

        active_tiles = 0
        mean_error = 0.0
        if self.equitile_config.track_tile_statistics:
            mean_error = sum(
                self._tile_stats.get(t.id, {}).get("error_mean", 0)
                for t in self.graph.all_tiles
            ) / len(self.graph.tiles)

            active_tiles = sum(
                1
                for t in self.graph.all_tiles
                if self._tile_stats.get(t.id, {}).get("error_mean", 0)
                > self.equitile_config.sparsity_threshold
            )

        return {
            "loss": loss.item(),
            "accuracy": self.compute_metrics(logits, y),
            "mean_error": mean_error,
            "active_tiles": active_tiles,
            "mode": "pc",
            "enhanced": True,
        }

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with all enhancements."""
        if self.equitile_config.mode == "ep":
            # For now fallback to base EP, or raise
            return super().train_step(x, y)

        # Call base implementation which calls _pc_inference (using overridden _relaxation_step)
        # and _pc_learning (overridden here)
        return super().train_step(x, y)

    def forward(self, x: Tensor, steps: Optional[int] = None) -> Tensor:
        """Forward pass with normalization."""
        # Must reimplement forward to include normalization (via overridden methods)
        # But base forward calls _init_activities and _relax (which calls _relaxation_step)
        # So base forward should work!
        # EXCEPT for the Batch Norm on out_activities at the end.

        batch, device = x.shape[0], x.device
        steps = steps if steps is not None else self.equitile_config.inference_steps
        input_proj = self.W_in(x)

        self._init_activities(input_proj, batch, device)

        self._relax(input_proj, steps)

        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1,
        )

        if self.equitile_config.use_batch_norm and "hidden" in self.batch_norms:
            out_activities = self.batch_norms["hidden"](out_activities)

        return self.W_out(out_activities)

    def get_stats(self) -> Dict:
        """Get enhanced statistics."""
        stats = super().get_stats()
        if self.config.track_tile_statistics:
            stats["tile_statistics"] = self._tile_stats.copy()
        return stats

    def summarize(self) -> str:
        """Get model summary."""
        return super().summarize() + " (Enhanced)"


# Factory function for backward compatibility
EnhancedEPConfig = EnhancedEquiTileConfig


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
    """Create an enhanced EquiTile model."""
    enhanced_config = EnhancedEquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        use_layer_norm=use_layer_norm,
        use_curriculum=use_curriculum,
        **kwargs,
    )

    return EnhancedEquiTile(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        input_dim=input_dim,
        output_dim=output_dim,
        enhanced_config=enhanced_config,
        **kwargs,
    )
