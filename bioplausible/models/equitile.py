"""
EquiTile: Adaptive Equilibrium Propagation with Predictive-Coding Dynamics
===========================================================================

A bio-plausible learning framework combining:
- Two-phase equilibrium propagation structure
- Predictive-coding relaxation for inference  
- Task-driven local weight updates (like TileEQ/ATPC)
- Learned per-tile importance for adaptive computation

Key Features
------------
- Two-phase relaxation (free + nudged) for bio-plausible credit assignment
- Local Hebbian weight updates driven by task errors
- No global backpropagation through the computational graph
- All updates use only tile-local information + neighbor errors
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BioModel, ModelConfig, register_model

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class EquiTileConfig:
    """Configuration for EquiTile."""
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 4

    # EP dynamics
    beta: float = 0.1
    inference_steps: int = 10
    step_size: float = 0.1
    lambda_error: float = 0.1

    # Learning
    learning_rate: float = 0.01
    importance_lr: float = 0.001

    # Adaptive computation
    sparsity_threshold: float = 0.01
    min_active_fraction: float = 0.1

    # Regularization
    importance_decay: float = 0.95
    weight_decay: float = 1e-4
    dropout: float = 0.1
    gradient_clip: float = 1.0


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


@dataclass
class EdgeParams:
    """Parameters for a directed edge."""
    src_id: int
    dst_id: int
    weight: Optional[Tensor] = None
    bias: Optional[Tensor] = None


class TileGraph:
    """Manages tile connectivity and state."""

    def __init__(self) -> None:
        self.tiles: Dict[int, TileState] = {}
        self.edges: Dict[Tuple[int, int], EdgeParams] = {}
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
                actual_neurons = min(neurons_per_tile, dim - tile_col * neurons_per_tile)

                tile = TileState(
                    id=tile_id,
                    neurons=actual_neurons,
                    layer_id=layer_idx,
                    pos_x=float(layer_idx) / max(1, total_layers - 1),
                    pos_y=(float(tile_col) / max(1, n_tiles - 1)) if n_tiles > 1 else 0.5,
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
        src = self.tiles[src_id]
        dst = self.tiles[dst_id]

        src.fwd_neighbors.append(dst_id)
        dst.bwd_neighbors.append(src_id)

        self.edges[(src_id, dst_id)] = EdgeParams(
            src_id=src_id,
            dst_id=dst_id,
            weight=torch.zeros(src.neurons, dst.neurons),
            bias=torch.zeros(dst.neurons),
        )

    @property
    def all_tiles(self) -> List[TileState]:
        return [self.tiles[i] for i in sorted(self.tiles.keys())]


@register_model("equitile")
class EquiTile(BioModel):
    """EquiTile: Adaptive Equilibrium Propagation.

    Combines two-phase EP structure with task-driven local learning.
    """

    algorithm_name = "EquiTile"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        neurons_per_tile: int,
        num_layers: int,
        tiles_per_layer: int,
        input_dim: int,
        output_dim: int,
        beta: float = 0.1,
        inference_steps: int = 10,
        step_size: float = 0.1,
        lambda_error: float = 0.1,
        learning_rate: float = 0.01,
        importance_lr: float = 0.001,
        sparsity_threshold: float = 0.01,
        min_active_fraction: float = 0.1,
        importance_decay: float = 0.95,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        gradient_clip: float = 1.0,
        activation: Literal["tanh", "relu", "gelu"] = "gelu",
        topology: Literal["layered", "custom"] = "layered",
        custom_edges: Optional[List[Tuple[int, int]]] = None,
        task_type: Literal["classification", "regression", "binary", "multilabel"] = "classification",
        **kwargs,
    ):
        if config is None:
            config = ModelConfig(
                name="equitile",
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[neurons_per_tile * tiles_per_layer] * (max(0, num_layers - 2)),
                learning_rate=learning_rate,
            )

        super().__init__(config, **kwargs)

        self.task_type = task_type
        self.config = EquiTileConfig(
            neurons_per_tile=neurons_per_tile,
            num_layers=num_layers,
            tiles_per_layer=tiles_per_layer,
            beta=beta,
            inference_steps=inference_steps,
            step_size=step_size,
            lambda_error=lambda_error,
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            sparsity_threshold=sparsity_threshold,
            min_active_fraction=min_active_fraction,
            importance_decay=importance_decay,
            weight_decay=weight_decay,
            dropout=dropout,
            gradient_clip=gradient_clip,
        )

        self.activation = self._get_activation(activation)
        self.graph = TileGraph()

        if topology == "layered":
            num_hidden = max(0, num_layers - 2)
            self.graph.build_layered(
                input_dim, output_dim,
                neurons_per_tile, num_hidden, tiles_per_layer
            )
        elif topology == "custom":
            if custom_edges is None:
                raise ValueError("custom_edges required")
            max_tile_id = max(max(src, dst) for src, dst in custom_edges)
            n_tiles = max_tile_id + 1
            self.graph.build_custom(n_tiles, neurons_per_tile, custom_edges, [0], [n_tiles-1])

        # I/O projections
        input_tile_dim = sum(self.graph.tiles[tid].neurons for tid in self.graph.input_tile_ids)
        output_tile_dim = sum(self.graph.tiles[tid].neurons for tid in self.graph.output_tile_ids)

        self.W_in = nn.Linear(input_dim, input_tile_dim)
        self.W_out = nn.Linear(output_tile_dim, output_dim)

        # Tile importance
        self.tile_importance = nn.Parameter(torch.ones(len(self.graph.tiles)))
        self.edge_importance = nn.Parameter(torch.ones(len(self.graph.edges)))

        # Optimizers
        self._optim_io = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=learning_rate,
        )
        self._optim_importance = torch.optim.Adam(
            [self.tile_importance, self.edge_importance],
            lr=importance_lr,
        )

        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._error_ema: Dict[int, float] = {}
        self._step_count = 0

        self._init_weights()

    def _get_activation(self, name: str):
        if name == "tanh":
            return torch.tanh
        elif name == "relu":
            return F.relu
        return F.gelu

    def _init_weights(self) -> None:
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for edge in self.graph.edges.values():
                if edge.weight is not None:
                    fan_in = edge.weight.shape[0]
                    std = math.sqrt(2.0 / fan_in)
                    edge.weight.normal_(0, std)
                if edge.bias is not None:
                    nn.init.zeros_(edge.bias)

            nn.init.kaiming_normal_(self.W_in.weight, mode='fan_in', nonlinearity='relu')
            if self.W_in.bias is not None:
                nn.init.zeros_(self.W_in.bias)
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
        return model

    def _apply_activation(self, x: Tensor) -> Tensor:
        return self._dropout(self.activation(x))

    def _compute_predictions(self, batch_size: int, device: torch.device) -> None:
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
        for tile in self.graph.all_tiles:
            if tile.activity is None:
                continue

            if tile.prediction is None:
                tile.error = tile.activity.clone()
            else:
                tile.error = tile.activity - tile.prediction

            err_norm = tile.error.norm(p=2, dim=-1).mean().item()
            self._error_ema[tile.id] = (
                self.config.importance_decay * self._error_ema.get(tile.id, 0.0)
                + (1 - self.config.importance_decay) * err_norm
            )

    def _relax(self, input_proj: Tensor, steps: int, output_nudge: Optional[Tensor] = None) -> None:
        batch_size = input_proj.shape[0]
        device = input_proj.device
        step_size = self.config.step_size

        for _ in range(steps):
            self._compute_predictions(batch_size, device)
            self._compute_errors()

            for i, tile in enumerate(self.graph.all_tiles):
                if tile.is_input:
                    idx = self.graph.input_tile_ids.index(tile.id)
                    start = idx * self.config.neurons_per_tile
                    tile.activity = input_proj[:, start:start + tile.neurons].clone()
                    continue

                if tile.error is None:
                    continue

                imp = torch.sigmoid(self.tile_importance[i]).item()
                grad = tile.error + self.config.lambda_error * tile.activity

                # Top-down modulation
                for dst_id in tile.fwd_neighbors:
                    dst = self.graph.tiles[dst_id]
                    edge = self.graph.edges.get((tile.id, dst_id))
                    if edge and edge.weight is not None and dst.error is not None:
                        grad = grad + dst.error @ edge.weight.T

                delta = step_size * imp * grad
                tile.activity = tile.activity - delta
                tile.activity = torch.clamp(tile.activity, -5.0, 5.0)

            if output_nudge is not None:
                for i, tile_id in enumerate(self.graph.output_tile_ids):
                    tile = self.graph.tiles[tile_id]
                    if tile.activity is not None:
                        start = i * self.config.neurons_per_tile
                        end = start + tile.neurons
                        if end <= output_nudge.shape[1]:
                            tile.activity = tile.activity + self.config.beta * output_nudge[:, start:end]
                            tile.activity = torch.clamp(tile.activity, -5.0, 5.0)

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Train with predictive-coding relaxation + task-driven local learning."""
        batch, device = x.shape[0], x.device
        self._step_count += 1

        input_proj = self.W_in(x)

        # Initialize
        for tile in self.graph.all_tiles:
            if tile.is_input:
                idx = self.graph.input_tile_ids.index(tile.id)
                start = idx * self.config.neurons_per_tile
                tile.activity = input_proj[:, start:start + tile.neurons].clone()
            else:
                tile.activity = torch.zeros(batch, tile.neurons, device=device)
            tile.prediction = None
            tile.error = None

        # === INFERENCE PHASE: Minimize prediction errors ===
        for _ in range(self.config.inference_steps):
            self._compute_predictions(batch, device)
            self._compute_errors()
            self._update_activities(input_proj)

        # === TASK-DRIVEN LEARNING ===
        # Compute output and loss
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1
        )
        logits = self.W_out(out_activities)

        if self.task_type == "regression":
            y_target = y.float()
            if y_target.dim() < logits.dim():
                y_target = y_target.unsqueeze(-1)
            loss = F.mse_loss(logits, y_target)
            output_delta = (logits - y_target) @ self.W_out.weight
        elif self.task_type == "binary":
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            output_delta = (logits.sigmoid() - y.float()).unsqueeze(-1) @ self.W_out.weight if y.dim() < logits.dim() else (logits.sigmoid() - y.float()) @ self.W_out.weight
        elif self.task_type == "multilabel":
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
                self.config.gradient_clip
            )
        self._optim_io.step()

        # === LOCAL HEBBIAN UPDATES FOR INTERNAL WEIGHTS ===
        # Backpropagate error layer by layer (local computation)
        tile_errors: Dict[int, Tensor] = {}

        # Output tiles get error from output_delta
        for i, tile_id in enumerate(self.graph.output_tile_ids):
            tile = self.graph.tiles[tile_id]
            start = i * self.config.neurons_per_tile
            end = start + tile.neurons
            tile_errors[tile_id] = output_delta[:, start:end].clone()

        # Hidden tiles: accumulate error from forward neighbors
        hidden_tiles = sorted(
            [t for t in self.graph.all_tiles if not t.is_output and not t.is_input],
            key=lambda t: -t.layer_id
        )
        for tile in hidden_tiles:
            error = torch.zeros_like(tile.activity)
            for fwd_id in tile.fwd_neighbors:
                if fwd_id not in tile_errors:
                    continue
                edge = self.graph.edges.get((tile.id, fwd_id))
                if edge and edge.weight is not None:
                    error = error + tile_errors[fwd_id] @ edge.weight.T
            tile_errors[tile.id] = error

        # Update internal weights with local Hebbian rule
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

                # Hebbian update: correlate pre-synaptic activity with post-synaptic error
                weight_update = imp * (src_act.T @ dst_err) / batch
                bias_update = imp * dst_err.mean(dim=0) / batch

                if edge.weight is not None:
                    edge.weight.data = edge.weight.data - lr * (
                        weight_update + self.config.weight_decay * edge.weight.data
                    )
                if edge.bias is not None:
                    edge.bias.data = edge.bias.data - lr * bias_update

        # Update importance
        self._update_importance()

        # Metrics
        with torch.no_grad():
            if self.task_type == "regression":
                mse = F.mse_loss(logits, y.float()).item()
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

        active_tiles = sum(
            1 for t in self.graph.all_tiles
            if self._error_ema.get(t.id, 0.0) > self.config.sparsity_threshold
        )

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "mean_error": sum(self._error_ema.get(t.id, 0.0) for t in self.graph.all_tiles) / len(self.graph.tiles),
            "active_tiles": active_tiles,
            "active_tiles_pct": active_tiles / len(self.graph.tiles) * 100,
        }

    def _update_activities(self, input_proj: Tensor) -> None:
        """Update tile activities to minimize prediction errors."""
        batch_size = input_proj.shape[0]
        step_size = self.config.step_size

        for i, tile in enumerate(self.graph.all_tiles):
            if tile.is_input:
                idx = self.graph.input_tile_ids.index(tile.id)
                start = idx * self.config.neurons_per_tile
                tile.activity = input_proj[:, start:start + tile.neurons].clone()
                continue

            if tile.error is None:
                continue

            imp = torch.sigmoid(self.tile_importance[i]).item()
            grad = tile.error + self.config.lambda_error * tile.activity

            # Top-down modulation from forward neighbors
            for dst_id in tile.fwd_neighbors:
                dst = self.graph.tiles[dst_id]
                edge = self.graph.edges.get((tile.id, dst_id))
                if edge and edge.weight is not None and dst.error is not None:
                    grad = grad + dst.error @ edge.weight.T

            delta = step_size * imp * grad
            tile.activity = tile.activity - delta
            tile.activity = torch.clamp(tile.activity, -5.0, 5.0)

    def _update_importance(self) -> None:
        self._optim_importance.zero_grad()

        tile_loss = torch.tensor(0.0, device=self.tile_importance.device)
        for i, tile in enumerate(self.graph.all_tiles):
            if tile.error is None:
                continue
            err_norm = tile.error.norm(p=2, dim=-1).mean()
            imp = torch.sigmoid(self.tile_importance[i])
            tile_loss = tile_loss + imp * err_norm.detach()

        sparsity_loss = 0.1 * torch.sum(torch.sigmoid(self.tile_importance))
        total_loss = tile_loss + sparsity_loss
        total_loss.backward()
        self._optim_importance.step()

    def forward(self, x: Tensor, steps: Optional[int] = None, return_states: bool = False) -> Tensor:
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

                tile.activity = tile.activity - self.config.step_size * imp * grad
                tile.activity = torch.clamp(tile.activity, -5.0, 5.0)

        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1
        )
        logits = self.W_out(out_activities)

        if return_states:
            states = {
                tile.id: {
                    "activity": tile.activity.clone() if tile.activity is not None else None,
                    "error": tile.error.clone() if tile.error is not None else None,
                }
                for tile in self.graph.all_tiles
            }
            return logits, states

        return logits

    def get_stats(self) -> Dict[str, float]:
        stats = super().get_stats()
        importances = torch.sigmoid(self.tile_importance).tolist()
        errors = [self._error_ema.get(t.id, 0.0) for t in self.graph.all_tiles]

        stats.update({
            "importance_mean": sum(importances) / len(importances),
            "importance_max": max(importances),
            "error_mean": sum(errors) / len(errors),
            "error_max": max(errors),
            "active_tiles": sum(1 for e in errors if e > self.config.sparsity_threshold),
            "total_tiles": len(self.graph.tiles),
            "total_edges": len(self.graph.edges),
        })
        return stats

    def summarize(self) -> str:
        return f"""============================================================
EquiTile: Adaptive Equilibrium Propagation
============================================================
Task Type: {self.task_type}
Architecture: {self.config.num_layers} layers, {self.config.tiles_per_layer} tiles/layer
Neurons per tile: {self.config.neurons_per_tile}
Total tiles: {len(self.graph.tiles)}
Total edges: {len(self.graph.edges)}
Total parameters: {sum(p.numel() for p in self.parameters()):,}

EP Hyperparameters:
  Beta (nudge): {self.config.beta}
  Inference steps: {self.config.inference_steps}
  Step size: {self.config.step_size}
  Learning rate: {self.config.learning_rate}
============================================================"""

    def get_state(self) -> Dict:
        edge_states = {
            f"{src}_{dst}": {
                "weight": edge.weight.cpu().numpy() if edge.weight is not None else None,
                "bias": edge.bias.cpu().numpy() if edge.bias is not None else None,
            }
            for (src, dst), edge in self.graph.edges.items()
        }

        return {
            "model_state_dict": self.state_dict(),
            "edge_states": edge_states,
            "task_type": self.task_type,
            "config": {
                "neurons_per_tile": self.config.neurons_per_tile,
                "num_layers": self.config.num_layers,
                "tiles_per_layer": self.config.tiles_per_layer,
            },
            "training": {"step_count": self._step_count, "error_ema": dict(self._error_ema)},
        }

    def load_state(self, state: Dict) -> None:
        self.load_state_dict(state["model_state_dict"], strict=False)
        if "edge_states" in state:
            with torch.no_grad():
                for key, edge_state in state["edge_states"].items():
                    src, dst = map(int, key.split("_"))
                    edge = self.graph.edges.get((src, dst))
                    if edge and edge_state["weight"] is not None:
                        edge.weight.copy_(torch.from_numpy(edge_state["weight"]))
                    if edge and edge_state["bias"] is not None:
                        edge.bias.copy_(torch.from_numpy(edge_state["bias"]))
        if "task_type" in state:
            self.task_type = state["task_type"]
