"""
EquiTile: Scalable Local-Learning Architecture with Tiled Substrates
====================================================================

A high-performance, scalable deep learning framework featuring:
- Tile-based parallel architecture for distributed training
- Local Hebbian weight updates (no global backpropagation tape)
- Learned tile importance for adaptive sparse computation
- Hardware-efficient design (GPU, TPU, edge accelerators)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Dict, List, Literal, Optional, Set, Tuple,
                    Union)

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.models.base import BioModel, ModelConfig, register_model

from .config import EquiTileConfig

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class TileState:
    """State for a single tile."""

    id: int
    neurons: int
    layer_id: int

    # Dynamic state
    activity: Optional[Tensor] = None
    prediction: Optional[Tensor] = None
    error: Optional[Tensor] = None

    # Metadata
    is_input: bool = False
    is_output: bool = False
    pos_x: float = 0.0
    pos_y: float = 0.0

    # Connectivity
    fwd_neighbors: List[int] = field(default_factory=list)
    bwd_neighbors: List[int] = field(default_factory=list)


class TileGraph:
    """Manages tile connectivity and state."""

    def __init__(self) -> None:
        self.tiles: Dict[int, TileState] = {}
        self.edges: List[Tuple[int, int]] = []
        self._edge_set: Set[Tuple[int, int]] = set()
        self.layer_ids: List[List[int]] = []
        self.input_tile_ids: List[int] = []
        self.output_tile_ids: List[int] = []

    def build_layered(
        self,
        input_dim: int,
        output_dim: int,
        neurons_per_tile: int,
        num_hidden_layers: int,
        tiles_per_layer: int = 1,
    ) -> None:
        hidden_dim = neurons_per_tile * tiles_per_layer
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        total_layers = len(dims)

        tile_id = 0

        for layer_idx, dim in enumerate(dims):
            n_tiles = math.ceil(dim / neurons_per_tile)
            layer_tile_ids: List[int] = []

            for tile_col in range(n_tiles):
                actual_neurons = min(
                    neurons_per_tile, dim - tile_col * neurons_per_tile
                )

                tile = TileState(
                    id=tile_id,
                    neurons=actual_neurons,
                    layer_id=layer_idx,
                    pos_x=float(layer_idx) / max(1, total_layers - 1),
                    pos_y=(
                        (float(tile_col) / max(1, n_tiles - 1)) if n_tiles > 1 else 0.5
                    ),
                    is_input=(layer_idx == 0),
                    is_output=(layer_idx == len(dims) - 1),
                )
                self.tiles[tile_id] = tile
                layer_tile_ids.append(tile_id)
                tile_id += 1

            self.layer_ids.append(layer_tile_ids)

        self.input_tile_ids = list(self.layer_ids[0])
        self.output_tile_ids = list(self.layer_ids[-1])

        for layer_idx in range(len(self.layer_ids) - 1):
            for src_id in self.layer_ids[layer_idx]:
                for dst_id in self.layer_ids[layer_idx + 1]:
                    self._add_edge(src_id, dst_id)

    def build_custom(
        self,
        n_tiles: int,
        neurons_per_tile: int,
        edges: List[Tuple[int, int]],
        input_ids: List[int],
        output_ids: List[int],
    ) -> None:
        """Build custom topology."""
        for tile_id in range(n_tiles):
            tile = TileState(
                id=tile_id,
                neurons=neurons_per_tile,
                layer_id=0,
                is_input=(tile_id in input_ids),
                is_output=(tile_id in output_ids),
            )
            self.tiles[tile_id] = tile

        self.input_tile_ids = list(input_ids)
        self.output_tile_ids = list(output_ids)

        for src_id, dst_id in edges:
            self._add_edge(src_id, dst_id)

    def _add_edge(self, src_id: int, dst_id: int) -> None:
        if (src_id, dst_id) in self._edge_set:
            return

        src = self.tiles[src_id]
        dst = self.tiles[dst_id]

        src.fwd_neighbors.append(dst_id)
        dst.bwd_neighbors.append(src_id)

        self.edges.append((src_id, dst_id))
        self._edge_set.add((src_id, dst_id))

    @property
    def all_tiles(self) -> List[TileState]:
        return [self.tiles[i] for i in sorted(self.tiles.keys())]


@register_model("equitile")
class EquiTile(BioModel):
    """EquiTile: Scalable Local-Learning Architecture."""

    algorithm_name = "EquiTile"

    def __init__(
        self,
        config: Optional[EquiTileConfig] = None,
        # Legacy/Flat arguments
        neurons_per_tile: int = 64,
        num_layers: int = 4,
        tiles_per_layer: int = 4,
        input_dim: int = 10,
        output_dim: int = 10,
        learning_rate: float = 0.01,
        mode: Literal["pc", "ep", "backprop"] = "pc",
        topology: Literal["layered", "custom"] = "layered",
        custom_edges: Optional[List[Tuple[int, int]]] = None,
        task_type: Literal[
            "classification", "regression", "binary", "multilabel"
        ] = "classification",
        activation: Literal["tanh", "relu", "gelu"] = "gelu",
        **kwargs,
    ) -> None:
        """Initialize EquiTile model.

        Parameters
        ----------
        config : EquiTileConfig, optional
            Configuration object. If provided, other args are ignored/merged.
        neurons_per_tile : int
            Number of neurons per tile.
        num_layers : int
            Number of layers (including input/output).
        tiles_per_layer : int
            Number of tiles per hidden layer.
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        learning_rate : float
            Learning rate.
        mode : str
            Training mode ('pc', 'ep', 'backprop').
        topology : str
            Topology type ('layered', 'custom').
        custom_edges : list of tuple, optional
            Edges for custom topology.
        task_type : str
            Task type.
        activation : str
            Activation function name.
        **kwargs
            Additional configuration parameters.
        """
        # 1. Handle Configuration
        if config is None:
            # Construct from args if config not provided
            config = EquiTileConfig(
                neurons_per_tile=neurons_per_tile,
                num_layers=num_layers,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                mode=mode,
                **kwargs,
            )

        self.equitile_config = config
        self.task_type = task_type
        self.activation_name = activation

        # 2. Initialize BioModel
        # Determine hidden dims for BioModel config (informational)
        hidden_dims = [config.neurons_per_tile * config.tiles_per_layer] * (
            max(0, config.num_layers - 2)
        )
        model_config = ModelConfig(
            name="equitile",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            learning_rate=config.learning_rate,
        )
        super().__init__(model_config)

        # 3. Build Graph
        self.graph = TileGraph()
        if topology == "layered":
            num_hidden = max(0, config.num_layers - 2)
            self.graph.build_layered(
                self.input_dim,
                self.output_dim,
                config.neurons_per_tile,
                num_hidden,
                config.tiles_per_layer,
            )
        elif topology == "custom":
            if custom_edges is None:
                raise ValueError("custom_edges required for custom topology")
            max_tile_id = max(max(src, dst) for src, dst in custom_edges)
            n_tiles = max_tile_id + 1
            self.graph.build_custom(
                n_tiles, config.neurons_per_tile, custom_edges, [0], [n_tiles - 1]
            )

        # 4. Initialize Parameters
        self._init_parameters(self.input_dim, self.output_dim)

        # 5. Initialize State
        self.activation = self._get_activation(activation)
        self._dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self._error_ema: Dict[int, float] = {}
        self._step_count = 0
        self._lr_scheduler = None
        self._lr_scheduler_type = None

        # 6. Setup Optimizers
        self._setup_optimizers()

    def get_config(self) -> EquiTileConfig:
        """Get the EquiTile configuration."""
        return self.equitile_config

    def _init_parameters(self, input_dim: int, output_dim: int) -> None:
        """Initialize model parameters."""
        # I/O projections
        input_tile_dim = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.input_tile_ids
        )
        output_tile_dim = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.output_tile_ids
        )

        self.W_in = nn.Linear(input_dim, input_tile_dim)
        self.W_out = nn.Linear(output_tile_dim, output_dim)

        # Edge parameters
        self.edge_weights = nn.ParameterDict()
        self.edge_biases = nn.ParameterDict()

        for src, dst in self.graph.edges:
            src_tile = self.graph.tiles[src]
            dst_tile = self.graph.tiles[dst]
            key = f"edge_{src}_{dst}"

            # Parameters will be initialized in _reset_weights
            weight = nn.Parameter(torch.empty(src_tile.neurons, dst_tile.neurons))
            bias = nn.Parameter(torch.empty(dst_tile.neurons))

            self.edge_weights[key] = weight
            self.edge_biases[key] = bias

        # Tile importance
        self.tile_importance = nn.Parameter(torch.ones(len(self.graph.tiles)))
        self.edge_importance = nn.Parameter(torch.ones(len(self.graph.edges)))

        self._reset_weights()

    def _reset_weights(self) -> None:
        """Reset weights to initial distribution."""
        with torch.no_grad():
            for key, weight in self.edge_weights.items():
                fan_in = weight.shape[0]
                std = math.sqrt(2.0 / fan_in)
                weight.normal_(0, std)

            for key, bias in self.edge_biases.items():
                nn.init.zeros_(bias)

            nn.init.kaiming_normal_(
                self.W_in.weight, mode="fan_in", nonlinearity="relu"
            )
            if self.W_in.bias is not None:
                nn.init.zeros_(self.W_in.bias)
            nn.init.xavier_normal_(self.W_out.weight, gain=1.0)
            if self.W_out.bias is not None:
                nn.init.zeros_(self.W_out.bias)

    def _setup_optimizers(self) -> None:
        """Initialize optimizers explicitly."""
        # I/O Optimizer
        self._optim_io = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=self.equitile_config.learning_rate,
        )

        # Importance Optimizer
        self._optim_importance = torch.optim.Adam(
            [self.tile_importance, self.edge_importance],
            lr=self.equitile_config.importance_lr,
        )

        # Full Optimizer (for backprop mode)
        # Note: We initialize this even if not in backprop mode to allow switching
        self._optim_full = torch.optim.Adam(
            self.parameters(), lr=self.equitile_config.learning_rate
        )

    def reset_optimizers(self) -> None:
        """Reset optimizers (e.g. after topology change)."""
        self._setup_optimizers()
        if self._lr_scheduler is not None:
            self.configure_lr_scheduler(
                scheduler_type=self._lr_scheduler_type,
                total_steps=self._total_steps,
                warmup_steps=self._warmup_steps,
            )

    def configure_lr_scheduler(
        self,
        scheduler_type: str = "cosine",
        total_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        warmup_steps: int = 100,
    ):
        """Configure learning rate scheduler."""
        self._lr_scheduler_type = scheduler_type

        if scheduler_type == "cosine":
            self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optim_io,
                T_max=total_steps - warmup_steps,
                eta_min=self.equitile_config.learning_rate * min_lr_ratio,
            )
        elif scheduler_type == "step":
            self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self._optim_io,
                step_size=total_steps // 5,
                gamma=0.5,
            )
        elif scheduler_type == "linear":
            self._lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self._optim_io,
                start_factor=1.0,
                end_factor=min_lr_ratio,
                total_iters=total_steps - warmup_steps,
            )

        self._warmup_steps = warmup_steps
        self._warmup_start_lr = self.equitile_config.learning_rate * 0.1
        self._total_steps = total_steps

    def step_lr_scheduler(self):
        """Step the learning rate scheduler."""
        if self._lr_scheduler is None:
            return

        # Handle warmup
        if hasattr(self, "_warmup_steps") and self._step_count < self._warmup_steps:
            warmup_progress = self._step_count / self._warmup_steps
            current_lr = (
                self._warmup_start_lr
                + (self.config.learning_rate - self._warmup_start_lr) * warmup_progress
            )

            for param_group in self._optim_io.param_groups:
                param_group["lr"] = current_lr
        else:
            self._lr_scheduler.step()

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        for param_group in self._optim_io.param_groups:
            return param_group["lr"]
        return self.equitile_config.learning_rate

    def _get_activation(self, name: str):
        if name == "tanh":
            return torch.tanh
        elif name == "relu":
            return F.relu
        return F.gelu

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        # nn.ParameterDict handles parameter movement automatically
        return model

    def _apply_activation(self, x: Tensor) -> Tensor:
        return self._dropout(self.activation(x))

    def _get_edge_params(
        self, src_id: int, dst_id: int
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        key = f"edge_{src_id}_{dst_id}"
        return self.edge_weights.get(key), self.edge_biases.get(key)

    def _compute_predictions(self, batch_size: int, device: torch.device) -> None:
        for tile in self.graph.all_tiles:
            if tile.is_input:
                continue

            pred = torch.zeros(batch_size, tile.neurons, device=device)

            for src_id in tile.bwd_neighbors:
                weight, bias = self._get_edge_params(src_id, tile.id)
                if weight is None:
                    continue

                src = self.graph.tiles[src_id]
                src_activity = (
                    src.activity
                    if src.activity is not None
                    else torch.zeros(batch_size, src.neurons, device=device)
                )
                pred = pred + self._apply_activation(src_activity) @ weight

                if bias is not None:
                    pred = pred + bias.unsqueeze(0)

            tile.prediction = pred

    def _compute_errors(self) -> None:
        for tile in self.graph.all_tiles:
            if tile.activity is None:
                continue

            if tile.prediction is None:
                tile.error = tile.activity.clone()
            else:
                tile.error = tile.activity - tile.prediction

            err_norm = tile.error.norm(p=2, dim=-1).mean().item()
            self._error_ema[tile.id] = (
                self.equitile_config.importance_decay
                * self._error_ema.get(tile.id, 0.0)
                + (1 - self.equitile_config.importance_decay) * err_norm
            )

    def _relax(
        self,
        input_proj: Tensor,
        steps: int,
        output_nudge: Optional[Tensor] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        """Run relaxation dynamics."""
        batch_size = input_proj.shape[0]
        step_size = self.equitile_config.step_size
        clamp = self.equitile_config.clamp_activities

        prev_activities = None
        for step in range(steps):
            self._compute_predictions(batch_size, input_proj.device)
            self._compute_errors()

            if tolerance is not None:
                prev_activities = {
                    tile.id: (
                        tile.activity.clone() if tile.activity is not None else None
                    )
                    for tile in self.graph.all_tiles
                }

            for i, tile in enumerate(self.graph.all_tiles):
                if tile.is_input:
                    idx = self.graph.input_tile_ids.index(tile.id)
                    start = idx * self.equitile_config.neurons_per_tile
                    tile.activity = input_proj[:, start : start + tile.neurons].clone()
                    continue

                if tile.error is None:
                    continue

                imp = torch.sigmoid(self.tile_importance[i]).item()
                grad = tile.error + self.equitile_config.lambda_error * tile.activity

                for dst_id in tile.fwd_neighbors:
                    dst = self.graph.tiles[dst_id]
                    weight, _ = self._get_edge_params(tile.id, dst_id)
                    if weight is not None and dst.error is not None:
                        grad = grad + dst.error @ weight.T

                delta = step_size * imp * grad
                tile.activity = tile.activity - delta

                if clamp:
                    tile.activity = torch.clamp(
                        tile.activity,
                        self.equitile_config.activity_clamp_min,
                        self.equitile_config.activity_clamp_max,
                    )

            if output_nudge is not None:
                for i, tile_id in enumerate(self.graph.output_tile_ids):
                    tile = self.graph.tiles[tile_id]
                    if tile.activity is not None:
                        start = i * self.equitile_config.neurons_per_tile
                        end = start + tile.neurons
                        if end <= output_nudge.shape[1]:
                            tile.activity = (
                                tile.activity
                                + self.equitile_config.beta * output_nudge[:, start:end]
                            )
                            if clamp:
                                tile.activity = torch.clamp(
                                    tile.activity,
                                    self.equitile_config.activity_clamp_min,
                                    self.equitile_config.activity_clamp_max,
                                )

            # Early stopping check
            if tolerance is not None and prev_activities is not None and step > 2:
                mean_change = 0.0
                count = 0
                for tile in self.graph.all_tiles:
                    if tile.is_input or prev_activities.get(tile.id) is None:
                        continue
                    if tile.activity is not None:
                        change = (
                            (tile.activity - prev_activities[tile.id])
                            .abs()
                            .mean()
                            .item()
                        )
                        mean_change += change
                        count += 1

                if count > 0:
                    mean_change /= count
                    if mean_change < tolerance:
                        break  # Converged

    def _compute_task_metrics(self, logits: Tensor, y: Tensor) -> float:
        """Compute task-specific accuracy metric."""
        with torch.no_grad():
            if self.task_type == "regression":
                # For regression, accuracy is R^2
                ss_res = ((y.float() - logits.squeeze()) ** 2).sum()
                ss_tot = ((y.float() - y.float().mean()) ** 2).sum()
                accuracy = 1 - (ss_res / (ss_tot + 1e-8))
            elif self.task_type == "binary":
                preds = (logits.sigmoid() > 0.5).long()
                accuracy = (preds.squeeze(-1) == y).float().mean().item()
            elif self.task_type == "multilabel":
                preds = (logits.sigmoid() > 0.5).long()
                accuracy = (preds == y).all(dim=-1).float().mean().item()
            else:
                accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
        return accuracy

    def _compute_metrics(self, logits: Tensor, y: Tensor) -> float:
        """Compute task-specific accuracy metric."""
        return self._compute_task_metrics(logits, y)

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with predictive-coding (PC) or equilibrium propagation (EP) mode."""
        self._step_count += 1
        if self.equitile_config.mode == "backprop":
            return self._train_step_backprop(x, y)
        elif self.equitile_config.mode == "ep":
            return self._train_step_ep(x, y)
        return self._train_step_pc(x, y)

    def _train_step_backprop(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train using standard backpropagation through time (BPTT)."""
        logits = self.forward(x, steps=self.equitile_config.inference_steps)
        loss = self._compute_loss(logits, y)

        self._optim_full.zero_grad()
        loss.backward()
        if self.equitile_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._optim_full.param_groups[0]["params"],
                self.equitile_config.gradient_clip,
            )
        self._optim_full.step()

        self._update_importance()
        accuracy = self._compute_metrics(logits, y)
        return {"loss": loss.item(), "accuracy": accuracy, "mode": "backprop"}

    def _train_step_pc(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with predictive-coding relaxation + task-driven local learning."""
        input_proj = self.W_in(x)
        batch = x.shape[0]

        # 1. Inference
        self._pc_inference(input_proj, batch, x.device)

        # 2. Learning
        return self._pc_learning(x, y, batch)

    def _init_activities(
        self,
        input_proj: Tensor,
        batch: int,
        device: torch.device,
        init_scale: float = 0.0,
    ) -> None:
        """Initialize tile activities, predictions, and errors."""
        for tile in self.graph.all_tiles:
            if tile.is_input:
                idx = self.graph.input_tile_ids.index(tile.id)
                start = idx * self.equitile_config.neurons_per_tile
                tile.activity = input_proj[:, start : start + tile.neurons].clone()
            else:
                tile.activity = (
                    torch.zeros(batch, tile.neurons, device=device) * init_scale
                    if init_scale != 0.0
                    else torch.zeros(batch, tile.neurons, device=device)
                )
            tile.prediction = None
            tile.error = None

    def _pc_inference(
        self, input_proj: Tensor, batch: int, device: torch.device
    ) -> None:
        """Run PC inference phase."""
        self._init_activities(input_proj, batch, device)
        self._relax(input_proj, self.equitile_config.inference_steps)

    def _pc_learning(self, x: Tensor, y: Tensor, batch: int) -> Dict[str, float]:
        """Run PC learning phase."""
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1,
        )
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

        # Local Updates
        self._apply_hebbian_updates(output_delta, batch)
        self._update_importance()

        return {
            "loss": loss.item(),
            "accuracy": self._compute_metrics(logits, y),
            "active_tiles": self._count_active_tiles(),
            "mode": "pc",
        }

    def _train_step_ep(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with strict two-phase Equilibrium Propagation."""
        batch = x.shape[0]
        input_proj = self.W_in(x)

        # Beta schedule
        beta = self.equitile_config.beta * (
            self.equitile_config.beta_anneal**self._step_count
        )

        # 1. Free Phase
        activities_free = self._ep_free_phase(input_proj, batch, x.device)

        # 2. Nudged Phase
        activities_nudged, loss, logits = self._ep_nudged_phase(
            input_proj, y, batch, x.device
        )

        # 3. Update
        self._ep_update(activities_free, activities_nudged, beta, batch)

        # Update I/O
        self._optim_io.zero_grad()
        loss.backward()
        if self.equitile_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.W_in.parameters()) + list(self.W_out.parameters()),
                self.equitile_config.gradient_clip,
            )
        self._optim_io.step()

        self._update_importance()

        return {
            "loss": loss.item(),
            "accuracy": self._compute_metrics(logits, y),
            "active_tiles": self._count_active_tiles(),
            "mode": "ep",
            "beta": beta,
        }

    def _ep_free_phase(
        self, input_proj: Tensor, batch: int, device: torch.device
    ) -> Dict[int, Tensor]:
        """Run EP free phase."""
        self._init_activities(
            input_proj, batch, device, init_scale=self.equitile_config.ep_init_scale
        )

        steps = (
            self.equitile_config.inference_steps_free
            or self.equitile_config.inference_steps
        )
        self._relax(
            input_proj, steps, tolerance=self.equitile_config.relaxation_tolerance
        )

        return {
            t.id: t.activity.clone()
            for t in self.graph.all_tiles
            if t.activity is not None
        }

    def _ep_nudged_phase(
        self, input_proj: Tensor, y: Tensor, batch: int, device: torch.device
    ) -> Tuple[Dict[int, Tensor], Tensor, Tensor]:
        """Run EP nudged phase."""
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1,
        )
        logits = self.W_out(out_activities)

        loss, output_nudge = self._compute_loss_and_nudge(logits, y)

        steps = (
            self.equitile_config.inference_steps_nudged
            or self.equitile_config.inference_steps
        )
        self._relax(
            input_proj,
            steps,
            output_nudge=output_nudge,
            tolerance=self.equitile_config.relaxation_tolerance,
        )

        activities_nudged = {
            t.id: t.activity.clone()
            for t in self.graph.all_tiles
            if t.activity is not None
        }
        return activities_nudged, loss, logits

    def _ep_update(
        self,
        free: Dict[int, Tensor],
        nudged: Dict[int, Tensor],
        beta: float,
        batch: int,
    ) -> None:
        """Apply contrastive Hebbian update."""
        lr = self.equitile_config.learning_rate
        with torch.no_grad():
            for edge_key in self.graph.edges:
                src_id, dst_id = edge_key
                weight, bias = self._get_edge_params(src_id, dst_id)
                if src_id not in free or dst_id not in free:
                    continue

                src_free, dst_free = self._apply_activation(
                    free[src_id]
                ), self._apply_activation(free[dst_id])
                src_nudged, dst_nudged = self._apply_activation(
                    nudged[src_id]
                ), self._apply_activation(nudged[dst_id])

                weight_update = (
                    (lr / beta)
                    * (src_free.T @ dst_free - src_nudged.T @ dst_nudged)
                    / batch
                )
                bias_update = (lr / beta) * (dst_free - dst_nudged).mean(dim=0) / batch

                if weight is not None:
                    weight.data = weight.data - weight_update.detach()
                    if self.equitile_config.weight_decay > 0:
                        weight.data = (
                            weight.data
                            - lr * self.equitile_config.weight_decay * weight.data
                        )
                if bias is not None:
                    bias.data = bias.data - bias_update.detach()

    def _get_loss_and_grad(self, logits: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute task-specific loss and gradient of loss w.r.t logits."""
        if self.task_type == "regression":
            y_target = y.float()
            if y_target.dim() < logits.dim():
                y_target = y_target.unsqueeze(-1)
            loss = F.mse_loss(logits, y_target)
            grad = logits - y_target
        elif self.task_type == "binary":
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            grad = (
                (logits.sigmoid() - y.float()).unsqueeze(-1)
                if y.dim() < logits.dim()
                else (logits.sigmoid() - y.float())
            )
        elif self.task_type == "multilabel":
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            grad = logits.sigmoid() - y.float()
        else:  # classification
            loss = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=-1)
            target_onehot = F.one_hot(y, self.output_dim).float().to(logits.device)
            grad = probs - target_onehot
        return loss, grad

    def _compute_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        loss, _ = self._get_loss_and_grad(logits, y)
        return loss

    def _compute_loss_and_delta(
        self, logits: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor]:
        loss, grad = self._get_loss_and_grad(logits, y)
        delta = grad @ self.W_out.weight
        return loss, delta

    def _compute_loss_and_nudge(
        self, logits: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor]:
        loss, delta = self._compute_loss_and_delta(logits, y)
        return loss, -delta

    def _apply_hebbian_updates(self, output_delta: Tensor, batch: int) -> None:
        """Apply local Hebbian updates."""
        tile_errors: Dict[int, Tensor] = {}
        for i, tile_id in enumerate(self.graph.output_tile_ids):
            tile = self.graph.tiles[tile_id]
            start = i * self.equitile_config.neurons_per_tile
            end = start + tile.neurons
            tile_errors[tile_id] = output_delta[:, start:end].clone()

        hidden_tiles = sorted(
            [t for t in self.graph.all_tiles if not t.is_output and not t.is_input],
            key=lambda t: -t.layer_id,
        )
        for tile in hidden_tiles:
            error = torch.zeros_like(tile.activity)
            for fwd_id in tile.fwd_neighbors:
                if fwd_id not in tile_errors:
                    continue
                weight, _ = self._get_edge_params(tile.id, fwd_id)
                if weight is not None:
                    error = error + tile_errors[fwd_id] @ weight.T
            tile_errors[tile.id] = error

        lr = self.equitile_config.learning_rate
        with torch.no_grad():
            for edge_idx, (src_id, dst_id) in enumerate(self.graph.edges):
                weight, bias = self._get_edge_params(src_id, dst_id)
                src, dst = self.graph.tiles[src_id], self.graph.tiles[dst_id]
                if src.activity is None or dst.id not in tile_errors:
                    continue

                imp = torch.sigmoid(self.edge_importance[edge_idx]).item()
                src_act = self._apply_activation(src.activity)
                dst_err = tile_errors[dst.id]

                weight_update = imp * (src_act.T @ dst_err) / batch
                bias_update = imp * dst_err.mean(dim=0) / batch

                if weight is not None:
                    weight.data = weight.data - lr * (
                        weight_update + self.equitile_config.weight_decay * weight.data
                    )
                if bias is not None:
                    bias.data = bias.data - lr * bias_update

    def _count_active_tiles(self) -> int:
        return sum(
            1
            for t in self.graph.all_tiles
            if self._error_ema.get(t.id, 0.0) > self.equitile_config.sparsity_threshold
        )

    def _update_importance(self) -> None:
        """Update tile and edge importance."""
        self._optim_importance.zero_grad()

        # 1. Tile Loss & Regularization
        tile_errors = []
        tile_indices = []
        for i, tile in enumerate(self.graph.all_tiles):
            if tile.error is not None:
                tile_errors.append(tile.error.norm(p=2, dim=-1).mean().detach())
                tile_indices.append(i)

        if tile_errors:
            tile_errors_t = torch.stack(tile_errors)
            tile_indices_t = torch.tensor(
                tile_indices, device=self.tile_importance.device
            )
            imps = torch.sigmoid(self.tile_importance[tile_indices_t])

            tile_loss = (imps * tile_errors_t).sum()
            reg_loss = (
                self.equitile_config.importance_reg_coef * ((imps - 0.5) ** 2)
            ).sum()
        else:
            tile_loss = torch.tensor(0.0, device=self.tile_importance.device)
            reg_loss = torch.tensor(0.0, device=self.tile_importance.device)

        # 2. Edge Loss & Regularization
        # self.edge_weights values are in same order as self.edge_importance
        edge_weights_list = list(self.edge_weights.values())
        if edge_weights_list:
            edge_norms_t = torch.stack([w.data.norm() for w in edge_weights_list])
            imps = torch.sigmoid(self.edge_importance)

            edge_loss = (imps * edge_norms_t).sum()
            edge_reg = (
                self.equitile_config.importance_reg_coef * ((imps - 0.5) ** 2)
            ).sum()
        else:
            edge_loss = torch.tensor(0.0, device=self.edge_importance.device)
            edge_reg = torch.tensor(0.0, device=self.edge_importance.device)

        # 3. Sparsity Loss (Applied to all)
        sparsity_loss = self.equitile_config.sparsity_penalty_coef * (
            torch.sigmoid(self.tile_importance).sum()
            + torch.sigmoid(self.edge_importance).sum()
        )

        total_loss = tile_loss + reg_loss + edge_loss + edge_reg + sparsity_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [self.tile_importance, self.edge_importance], max_norm=1.0
        )
        self._optim_importance.step()

    def forward(
        self, x: Tensor, steps: Optional[int] = None, return_states: bool = False
    ) -> Tensor:
        """Forward pass."""
        batch, device = x.shape[0], x.device
        steps = steps if steps is not None else self.equitile_config.inference_steps
        input_proj = self.W_in(x)

        # Initialize (same as inference)
        self._init_activities(input_proj, batch, device)

        self._relax(input_proj, steps)

        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1,
        )
        logits = self.W_out(out_activities)

        if return_states:
            states = {
                t.id: {
                    "activity": t.activity.clone() if t.activity is not None else None,
                    "error": t.error.clone() if t.error is not None else None,
                }
                for t in self.graph.all_tiles
            }
            return logits, states
        return logits

    def get_stats(self) -> Dict[str, float]:
        stats = super().get_stats()
        importances = torch.sigmoid(self.tile_importance).tolist()
        errors = [self._error_ema.get(t.id, 0.0) for t in self.graph.all_tiles]
        stats.update(
            {
                "importance_mean": sum(importances) / len(importances),
                "importance_max": max(importances),
                "error_mean": sum(errors) / len(errors),
                "error_max": max(errors),
                "active_tiles": sum(
                    1 for e in errors if e > self.equitile_config.sparsity_threshold
                ),
                "total_tiles": len(self.graph.tiles),
                "total_edges": len(self.graph.edges),
            }
        )
        return stats

    def summarize(self) -> str:
        return f"EquiTile(mode={self.equitile_config.mode}, layers={self.equitile_config.num_layers})"

    def get_state(self) -> Dict:
        """Get complete model state for checkpointing."""
        state = {
            "model_state_dict": self.state_dict(),
            "task_type": self.task_type,
            "config": self.equitile_config,
            "training": {
                "step_count": self._step_count,
                "error_ema": dict(self._error_ema),
                "warmup_steps": getattr(self, "_warmup_steps", 0),
                "total_steps": getattr(self, "_total_steps", 0),
            },
        }

        # Save optimizers if initialized
        if hasattr(self, "_optim_io"):
            state["optim_io"] = self._optim_io.state_dict()
        if hasattr(self, "_optim_importance"):
            state["optim_importance"] = self._optim_importance.state_dict()
        if hasattr(self, "_optim_full"):
            state["optim_full"] = self._optim_full.state_dict()

        # Save scheduler
        if self._lr_scheduler is not None:
            state["lr_scheduler"] = self._lr_scheduler.state_dict()
            state["lr_scheduler_type"] = self._lr_scheduler_type

        return state

    def load_state(self, state: Dict) -> None:
        """Load model state from checkpoint."""
        self.load_state_dict(state["model_state_dict"], strict=False)

        if "training" in state:
            self._step_count = state["training"].get("step_count", 0)
            self._error_ema = state["training"].get("error_ema", {})
            self._warmup_steps = state["training"].get("warmup_steps", 100)
            self._total_steps = state["training"].get("total_steps", 1000)

        # Restore optimizers (assuming they are initialized)
        if "optim_io" in state and hasattr(self, "_optim_io"):
            self._optim_io.load_state_dict(state["optim_io"])
        if "optim_importance" in state and hasattr(self, "_optim_importance"):
            self._optim_importance.load_state_dict(state["optim_importance"])
        if "optim_full" in state and hasattr(self, "_optim_full"):
            self._optim_full.load_state_dict(state["optim_full"])

        # Restore Scheduler
        if "lr_scheduler" in state and "lr_scheduler_type" in state:
            scheduler_type = state["lr_scheduler_type"]
            # We must re-configure the scheduler to load its state
            # Use sensible defaults or values from state if available
            total_steps = getattr(self, "_total_steps", 1000)
            warmup_steps = getattr(self, "_warmup_steps", 100)

            self.configure_lr_scheduler(
                scheduler_type=scheduler_type,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            try:
                self._lr_scheduler.load_state_dict(state["lr_scheduler"])
            except Exception:
                pass

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Save model checkpoint to disk."""
        state = self.get_state()
        if metadata:
            state["metadata"] = metadata
        torch.save(state, path)

    def load_checkpoint(
        self,
        path: str,
        device: Optional[torch.device] = None,
        load_optimizer: bool = True,
    ) -> Optional[Dict]:
        """Load model checkpoint from disk."""
        if device is None:
            device = next(self.parameters()).device

        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=device, weights_only=False)

        self.load_state(state)
        return state.get("metadata")


@register_model("equitile_ep")
class EquiTileEP(EquiTile):
    """EquiTile with strict Equilibrium Propagation learning."""

    algorithm_name = "EquiTileEP"

    def __init__(self, *args, beta: float = 0.1, **kwargs):
        kwargs["mode"] = "ep"
        kwargs["beta"] = beta
        super().__init__(*args, **kwargs)
