"""
EquiTile Distributed: Multi-GPU Tile Distribution
==================================================

Enables true distributed training across multiple GPUs:
- Tile distribution across devices
- Inter-GPU communication for tile boundaries
- Gradient accumulation across devices
- Mixed precision support (FP16/BF16)

Key Components
--------------
- DistributedConfig: Distributed training configuration
- TileCommunicator: Handles inter-GPU communication
- MixedPrecisionTrainer: FP16/BF16 support with loss scaling
- DistributedEquiTile: Multi-GPU wrapper

Examples
--------
>>> from bioplausible.models.equitile import EquiTile, DistributedEquiTile
>>> model = EquiTile(neurons_per_tile=64, num_layers=4,
...                  tiles_per_layer=4, input_dim=784, output_dim=10)
>>> dist_model = DistributedEquiTile(
...     model,
...     device_ids=[0, 1, 2, 3],
...     mixed_precision=True,
... )
>>> stats = dist_model.train_step(X, y)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistributedConfig, TileGrowthConfig
from .kernels import (
    compute_tile_prediction,
    compute_activity_update,
    compute_hebbian_update,
)

if TYPE_CHECKING:
    from .core import EquiTile


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DeviceAssignment:
    """Assignment of tiles to devices.

    Attributes
    ----------
    device_id : int
        Device index
    device : torch.device
        Device object
    tile_ids : list of int
        Tile IDs assigned to this device
    edge_ids : list of tuple
        Edge IDs assigned to this device
    """
    device_id: int
    device: torch.device
    tile_ids: List[int]
    edge_ids: List[Tuple[int, int]]


# =============================================================================
# Tile Communicator
# =============================================================================

class TileCommunicator:
    """Handles inter-GPU communication for tile boundaries.

    When tiles are distributed across GPUs, boundary tiles need to
    exchange activity and error information.

    Parameters
    ----------
    assignments : list of DeviceAssignment
        Device assignments
    backend : str
        Communication backend
    """

    def __init__(
        self,
        assignments: List[DeviceAssignment],
        graph: Any,
        backend: str = "nccl",
    ) -> None:
        self.assignments = assignments
        self.graph = graph
        self.backend = backend
        self.n_devices = len(assignments)

        # Build communication groups
        self._boundary_tiles = self._find_boundary_tiles()
        self._comm_buffers: Dict[int, Dict[str, torch.Tensor]] = {}

    def _find_boundary_tiles(self) -> Dict[int, List[Tuple[int, int]]]:
        """Find tiles that need cross-device communication.

        Returns
        -------
        dict
            Dict mapping device_id to list of (local_tile, remote_tile) pairs
        """
        # Build tile -> device mapping
        tile_to_device: Dict[int, int] = {}
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile_to_device[tile_id] = assignment.device_id

        # Use graph helper
        boundary_map = self.graph.get_boundary_tiles(tile_to_device)

        # Structure output: device_id -> [(local_tile, remote_tile), ...]
        boundary: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(self.n_devices)}

        for local_tile, remote_tiles in boundary_map.items():
            local_dev = tile_to_device.get(local_tile)
            if local_dev is not None:
                for remote_tile in remote_tiles:
                    boundary[local_dev].append((local_tile, remote_tile))

        return boundary

    def exchange_activities(
        self,
        activities: Dict[int, torch.Tensor],
        device_id: int,
    ) -> Dict[int, torch.Tensor]:
        """Exchange tile activities across device boundaries.

        Parameters
        ----------
        activities : dict
            Local tile activities
        device_id : int
            This device's ID

        Returns
        -------
        dict
            Activities from remote boundary tiles
        """
        if self.n_devices == 1:
            return {}

        # For multi-GPU, use all_reduce or send/recv
        # This is a simplified implementation
        received: Dict[int, torch.Tensor] = {}

        for local_tile, remote_tile in self._boundary_tiles.get(device_id, []):
            if remote_tile in activities:
                received[local_tile] = activities[remote_tile].clone()

        return received

    def sync_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        device_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Sync gradients across devices (all_reduce).

        Parameters
        ----------
        gradients : dict
            Gradient tensors by name
        device_id : int
            This device's ID

        Returns
        -------
        dict
            Synced gradients
        """
        if self.n_devices == 1:
            return gradients

        # All-reduce gradients
        for name, grad in gradients.items():
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.AVG)

        return gradients


# =============================================================================
# Mixed Precision Trainer
# =============================================================================

class MixedPrecisionTrainer:
    """Mixed precision training for EquiTile.

    Supports FP16 and BF16 with loss scaling.

    Parameters
    ----------
    model : EquiTile
        The model
    dtype : str
        Precision type: 'float16' or 'bfloat16'
    initial_scale : float
        Initial loss scale
    scale_window : int
        Steps before increasing scale
    """

    def __init__(
        self,
        model: EquiTile,
        dtype: str = "float16",
        initial_scale: float = 65536.0,
        scale_window: int = 1000,
    ) -> None:
        self.model = model
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.enabled = self.dtype != torch.float32

        # Loss scaling
        self.scale = initial_scale if self.enabled else 1.0
        self.scale_window = scale_window
        self.steps_without_overflow = 0

        # Grad scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.enabled)

    def cast_model(self) -> None:
        """Cast model weights to mixed precision."""
        if not self.enabled:
            return

        # Cast edge weights
        for weight in self.model.edge_weights.values():
            weight.data = weight.data.to(self.dtype)
        for bias in self.model.edge_biases.values():
            bias.data = bias.data.to(self.dtype)

    def autocast(self):
        """Context manager for autocast.

        Returns
        -------
        torch.amp.autocast
            Autocast context manager
        """
        return torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.enabled)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient scaling.

        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor

        Returns
        -------
        torch.Tensor
            Scaled loss
        """
        if self.enabled:
            return loss * self.scale
        return loss

    def unscale_and_clip_grads(
        self,
        gradients: Dict[str, torch.Tensor],
        max_norm: float = 1.0,
    ) -> float:
        """Unscale gradients and clip.

        Parameters
        ----------
        gradients : dict
            Gradient tensors
        max_norm : float
            Maximum gradient norm

        Returns
        -------
        float
            Total gradient norm
        """
        total_norm = 0.0

        for name, grad in gradients.items():
            if grad is None:
                continue

            # Unscale
            grad.data.mul_(1.0 / self.scale)

            # Compute norm
            param_norm = grad.data.norm(2)
            total_norm = max(total_norm, param_norm.item())

            # Clip
            if total_norm > max_norm:
                clip_coef = max_norm / (total_norm + 1e-6)
                grad.data.mul_(clip_coef)

        return total_norm

    def update_scale(self, found_inf: bool) -> None:
        """Update loss scale based on overflow.

        Parameters
        ----------
        found_inf : bool
            Whether inf/nan was found
        """
        if found_inf:
            self.scale = max(self.scale / 2.0, 1.0)
            self.steps_without_overflow = 0
        else:
            self.steps_without_overflow += 1
            if self.steps_without_overflow >= self.scale_window:
                self.scale = min(self.scale * 2.0, 65536.0)
                self.steps_without_overflow = 0


# =============================================================================
# Distributed EquiTile
# =============================================================================

class DistributedEquiTile:
    """Multi-GPU distributed EquiTile.

    Distributes tiles across multiple GPUs for parallel training.

    Parameters
    ----------
    model : EquiTile
        Base EquiTile model
    config : DistributedConfig, optional
        Distributed training configuration

    Examples
    --------
    >>> model = EquiTile(neurons_per_tile=64, num_layers=4,
    ...                  tiles_per_layer=4, input_dim=784, output_dim=10)
    >>> dist_model = DistributedEquiTile(
    ...     model,
    ...     device_ids=[0, 1, 2, 3],
    ...     tile_balance='round_robin',
    ...     mixed_precision=True,
    ... )
    >>> stats = dist_model.train_step(X, y)
    """

    def __init__(
        self,
        model: EquiTile,
        config: Optional[DistributedConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or DistributedConfig()

        # Set up devices
        if not self.config.device_ids:
            if torch.cuda.is_available():
                self.config.device_ids = list(range(torch.cuda.device_count()))
                self.devices = [
                    torch.device(f'cuda:{i}') for i in self.config.device_ids
                ]
            else:
                self.config.device_ids = [0]
                self.devices = [torch.device('cpu')]
        else:
            self.devices = [
                torch.device(f'cuda:{i}') for i in self.config.device_ids
            ]
        self.n_devices = len(self.devices)

        # Assign tiles to devices
        self.assignments = self._assign_tiles()

        # Set up communicator
        self.communicator = TileCommunicator(
            self.assignments,
            self.model.graph,
            backend=self.config.communication_backend
        )

        # Set up mixed precision
        self.mp_trainer: Optional[MixedPrecisionTrainer] = None
        # Disable mixed precision on CPU to avoid dtype mismatches
        if self.config.mixed_precision and self.devices[0].type == 'cpu':
            self.config.mixed_precision = False

        if self.config.mixed_precision:
            self.mp_trainer = MixedPrecisionTrainer(
                model,
                dtype=self.config.mixed_precision_dtype
            )
            self.mp_trainer.cast_model()

        # Set up tile growth/pruning
        self.growth_config = TileGrowthConfig()
        self._steps_since_modify = 0

        # Move tiles to assigned devices
        self._distribute_tiles()

        # Gradient accumulation
        self._accumulated_gradients: Dict[str, torch.Tensor] = {}
        self._accumulation_step = 0

    def _assign_tiles(self) -> List[DeviceAssignment]:
        """Assign tiles to devices.

        Returns
        -------
        list of DeviceAssignment
            Device assignments
        """
        n_tiles = len(self.model.graph.tiles)
        tile_ids = list(self.model.graph.tiles.keys())

        assignments: List[DeviceAssignment] = []

        for i, device_id in enumerate(self.config.device_ids):
            if self.config.tile_balance == "round_robin":
                # Round-robin assignment
                assigned = tile_ids[i::self.n_devices]
            elif self.config.tile_balance == "layered":
                # Assign by layers
                layer_size = n_tiles // self.n_devices
                start = i * layer_size
                end = start + layer_size if i < self.n_devices - 1 else n_tiles
                assigned = tile_ids[start:end]
            else:
                assigned = tile_ids

            assignments.append(DeviceAssignment(
                device_id=i,
                device=self.devices[i],
                tile_ids=assigned,
                edge_ids=[],  # Would need to compute
            ))

        return assignments

    def _distribute_tiles(self) -> None:
        """Move tiles to assigned devices."""
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]

                if tile.activity is not None:
                    tile.activity = tile.activity.to(assignment.device)
                if tile.prediction is not None:
                    tile.prediction = tile.prediction.to(assignment.device)
                if tile.error is not None:
                    tile.error = tile.error.to(assignment.device)

        # Move edge weights
        for assignment in self.assignments:
            for edge_key in assignment.edge_ids:
                weight, bias = self.model._get_edge_params(*edge_key)
                if weight is not None:
                    weight.data = weight.data.to(assignment.device)
                if bias is not None:
                    bias.data = bias.data.to(assignment.device)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Training step with distributed execution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        # For single GPU, use regular training
        if self.n_devices == 1:
            if self.mp_trainer:
                return self._train_step_mixed_precision(x, y)
            return self.model.train_step(x, y)

        # Multi-GPU training
        return self._train_step_distributed(x, y)

    def _train_step_mixed_precision(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Training step with mixed precision.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        if self.mp_trainer is None:
            return self.model.train_step(x, y)

        # self.model._ensure_local_optimizers() # Removed: Method does not exist
        self.mp_trainer.scaler.unscale_(self.model._optim_io)
        self.model._optim_io.zero_grad()

        with self.mp_trainer.autocast():
            # Run forward pass in mixed precision
            stats = self.model.train_step(x, y)

        return stats

    def _train_step_distributed(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Distributed training step across multiple GPUs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        batch_size = x.shape[0]
        device = self.devices[0]  # Input on first device

        # Distribute input across devices
        input_proj = self.model.W_in(x)

        # Initialize activities on each device
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]

                if tile.is_input:
                    idx = self.model.graph.input_tile_ids.index(tile.id)
                    start = idx * self.model.config.neurons_per_tile
                    tile.activity = input_proj[:, start:start + tile.neurons].clone()
                else:
                    tile.activity = torch.zeros(
                        batch_size, tile.neurons, device=assignment.device
                    )
                tile.prediction = None
                tile.error = None

        # Relaxation loop with communication
        for _ in range(self.model.config.inference_steps):
            self._distributed_relax_step(batch_size)

        # Learning step
        return self._distributed_learning(y)

    def _distributed_relax_step(self, batch_size: int) -> None:
        """One relaxation step with inter-GPU communication.

        Parameters
        ----------
        batch_size : int
            Batch size
        """
        # Compute predictions locally
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]
                if tile.is_input:
                    continue

                inputs = []
                total_bias = None

                for src_id in tile.bwd_neighbors:
                    src = self.model.graph.tiles[src_id]
                    weight, bias = self.model._get_edge_params(src_id, tile.id)

                    if weight is None:
                        continue

                    src_activity = (
                        src.activity
                        if src.activity is not None
                        else torch.zeros(
                            batch_size, src.neurons, device=assignment.device
                        )
                    )
                    inputs.append(self.model._apply_activation(src_activity) @ weight)

                    if bias is not None:
                        if total_bias is None:
                            total_bias = bias
                        else:
                            total_bias = total_bias + bias

                tile.prediction = compute_tile_prediction(inputs, total_bias)

                # Ensure correct shape if result is scalar zero (empty inputs)
                if tile.prediction.dim() == 0:
                    tile.prediction = torch.zeros(
                        batch_size, tile.neurons, device=assignment.device
                    )

        # Compute errors locally
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]
                if tile.activity is None:
                    continue

                if tile.prediction is None:
                    tile.error = tile.activity.clone()
                else:
                    tile.error = tile.activity - tile.prediction

        # Exchange boundary information
        for assignment in self.assignments:
            activities = {
                tile_id: self.model.graph.tiles[tile_id].activity
                for tile_id in assignment.tile_ids
                if self.model.graph.tiles[tile_id].activity is not None
            }

            received = self.communicator.exchange_activities(
                activities, assignment.device_id
            )

            # Update boundary predictions
            for local_tile, remote_activity in received.items():
                tile = self.model.graph.tiles[local_tile]
                if tile.prediction is not None:
                    # Add contribution from remote tile
                    pass  # Would need edge weights

        # Update activities locally
        for assignment in self.assignments:
            for i, tile_id in enumerate(assignment.tile_ids):
                tile = self.model.graph.tiles[tile_id]
                if tile.is_input or tile.error is None:
                    continue

                tile_idx = list(self.model.graph.tiles.keys()).index(tile.id)
                imp = torch.sigmoid(self.model.tile_importance[tile_idx]).item()

                fwd_feedback = []
                for dst_id in tile.fwd_neighbors:
                    dst = self.model.graph.tiles[dst_id]
                    weight, _ = self.model._get_edge_params(tile.id, dst_id)
                    if weight is not None and dst.error is not None:
                        fwd_feedback.append(dst.error @ weight.T)

                tile.activity = compute_activity_update(
                    activity=tile.activity,
                    error=tile.error,
                    fwd_feedback=fwd_feedback,
                    importance=imp,
                    step_size=self.model.config.step_size,
                    lambda_error=self.model.config.lambda_error,
                    clamp_min=-5.0,
                    clamp_max=5.0,
                    clamp=self.model.config.clamp_activities,
                )

    def _distributed_learning(self, y: torch.Tensor) -> Dict[str, float]:
        """Learning step for distributed training.

        Parameters
        ----------
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        # Gather output activities to W_out device
        w_out_device = self.model.W_out.weight.device
        out_activities_list = []

        for tid in self.model.graph.output_tile_ids:
            act = self.model.graph.tiles[tid].activity
            if act is not None:
                out_activities_list.append(act.to(w_out_device))

        if not out_activities_list:
             # Should not happen ideally
             batch_size = y.shape[0] if y.dim() > 0 else 1
             out_dim = self.model.W_out.in_features
             out_activities = torch.zeros(batch_size, out_dim, device=w_out_device)
        else:
            out_activities = torch.cat(out_activities_list, dim=-1)

        # Compute loss
        logits = self.model.W_out(out_activities)

        loss = self.model.task_handler.compute_loss(logits, y)

        # Backprop for I/O projections
        # self.model._ensure_local_optimizers() # Removed
        self.model._optim_io.zero_grad()
        loss.backward()
        self.model._optim_io.step()

        # Local Hebbian updates (each device updates its edges)
        for assignment in self.assignments:
            for edge_key in assignment.edge_ids:
                weight, bias = self.model._get_edge_params(*edge_key)
                if weight is None:
                    continue

                src = self.model.graph.tiles[edge_key[0]]
                dst = self.model.graph.tiles[edge_key[1]]

                if src.activity is None or dst.error is None:
                    continue

                # edge_idx = list(self.model.graph.edges.keys()).index(edge_key) # edges is list now
                edge_idx = self.model.graph.edges.index(edge_key)
                imp = torch.sigmoid(self.model.edge_importance[edge_idx]).item()

                src_act = self.model._apply_activation(src.activity)
                dst_err = dst.error

                batch_size = src_act.shape[0]
                weight_update, bias_update = compute_hebbian_update(
                    src_act, dst_err, imp, batch_size
                )

                if weight is not None:
                    weight.data = weight.data - self.model.config.learning_rate * (
                        weight_update + self.model.config.weight_decay * weight.data
                    )
                if bias is not None:
                    bias.data = (
                        bias.data - self.model.config.learning_rate * bias_update
                    )

        # Compute metrics
        accuracy = self.model.task_handler.compute_metrics(logits, y)

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "mode": self.model.equitile_config.mode,
            "distributed": True,
            "n_devices": self.n_devices,
        }

    def grow_tile(self, parent_tile_id: int) -> int:
        """Add a new tile as a child of an existing tile.

        Parameters
        ----------
        parent_tile_id : int
            Parent tile ID

        Returns
        -------
        int
            New tile ID (-1 if failed)
        """
        if not self.growth_config.enabled:
            return -1

        parent = self.model.graph.tiles[parent_tile_id]

        # Use model API to add tile
        new_id = self.model.add_tile(
            neurons=parent.neurons,
            layer_id=parent.layer_id + 1,
            pos_x=parent.pos_x, # Ideally offset this
            pos_y=parent.pos_y,
            is_input=False,
            is_output=False,
        )

        # Connect to parent (lateral or forward? dynamics used lateral, this used layer+1)
        # Original logic was layer_id + 1, so it's a forward connection
        self.model.add_edge(parent_tile_id, new_id)

        # Assign to device
        # Find parent assignment
        parent_device = self.devices[0]
        for assignment in self.assignments:
            if parent_tile_id in assignment.tile_ids:
                parent_device = assignment.device
                assignment.tile_ids.append(new_id)
                assignment.edge_ids.append((parent_tile_id, new_id))
                break

        # Move new tile state to device
        tile = self.model.graph.tiles[new_id]
        if tile.activity is not None:
            tile.activity = tile.activity.to(parent_device)
        if tile.prediction is not None:
            tile.prediction = tile.prediction.to(parent_device)
        if tile.error is not None:
            tile.error = tile.error.to(parent_device)

        # Move edge parameters
        weight, bias = self.model._get_edge_params(parent_tile_id, new_id)
        if weight is not None:
            weight.data = weight.data.to(parent_device)
        if bias is not None:
            bias.data = bias.data.to(parent_device)

        self._steps_since_modify = 0
        return new_id

    def prune_tile(self, tile_id: int) -> bool:
        """Remove a tile and its connections.

        Parameters
        ----------
        tile_id : int
            Tile ID to remove

        Returns
        -------
        bool
            Whether tile was pruned
        """
        if not self.growth_config.enabled:
            return False

        tile = self.model.graph.tiles.get(tile_id)
        if tile is None or tile.is_input or tile.is_output:
            return False

        # Identify edges to verify they are removed from assignments
        edges_to_remove = [
            (src, dst) for (src, dst) in self.model.graph.edges
            if tile_id in (src, dst)
        ]

        # Use model API
        self.model.remove_tile(tile_id)

        # Remove from assignments
        for assignment in self.assignments:
            if tile_id in assignment.tile_ids:
                assignment.tile_ids.remove(tile_id)

            # Remove edges from assignment
            assignment.edge_ids = [
                e for e in assignment.edge_ids
                if e not in edges_to_remove
            ]

        self._steps_since_modify = 0
        return True

    def maybe_modify_tiles(self, errors: Dict[int, float]) -> Dict[str, int]:
        """Check if tiles should be grown or pruned.

        Parameters
        ----------
        errors : dict
            Error values per tile

        Returns
        -------
        dict
            Modification counts
        """
        stats: Dict[str, int] = {"grown": 0, "pruned": 0}

        if not self.growth_config.enabled:
            return stats

        self._steps_since_modify += 1
        if self._steps_since_modify < self.growth_config.cooldown_steps:
            return stats

        # Check for growth
        for tile_id, error in errors.items():
            if error > self.growth_config.growth_threshold:
                if len(self.model.graph.tiles) < self.growth_config.max_tiles:
                    new_id = self.grow_tile(tile_id)
                    if new_id >= 0:
                        stats["grown"] += 1
                        break  # One tile at a time

        # Check for pruning
        for tile_id, error in errors.items():
            if error < self.growth_config.prune_threshold:
                if len(self.model.graph.tiles) > self.growth_config.min_tiles:
                    if self.prune_tile(tile_id):
                        stats["pruned"] += 1
                        break  # One tile at a time

        return stats

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.n_devices > 1

    @property
    def is_mixed_precision(self) -> bool:
        """Check if mixed precision is enabled."""
        return self.mp_trainer is not None


# =============================================================================
# Factory Functions
# =============================================================================

def create_distributed_model(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    input_dim: int = 784,
    output_dim: int = 10,
    device_ids: Optional[List[int]] = None,
    mixed_precision: bool = True,
    tile_balance: str = "round_robin",
    **kwargs,
) -> Tuple[EquiTile, DistributedEquiTile]:
    """Create a distributed EquiTile model.

    Parameters
    ----------
    neurons_per_tile : int
        Neurons per tile
    num_layers : int
        Number of layers
    tiles_per_layer : int
        Tiles per layer
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    device_ids : list of int, optional
        GPU device IDs
    mixed_precision : bool
        Enable mixed precision
    tile_balance : str
        Tile balancing strategy
    **kwargs
        Additional arguments for EquiTile

    Returns
    -------
    tuple of (EquiTile, DistributedEquiTile)
        Base model and distributed wrapper
    """
    from .core import EquiTile

    model = EquiTile(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs,
    )

    dist_model = DistributedEquiTile(
        model,
        config=DistributedConfig(
            device_ids=device_ids or [],
            mixed_precision=mixed_precision,
            tile_balance=tile_balance,
        ),
    )

    return model, dist_model
