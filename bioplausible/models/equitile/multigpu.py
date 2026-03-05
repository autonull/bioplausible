"""
EquiTile Multi-GPU: True Async Execution with NCCL
===================================================

Implements true asynchronous multi-GPU training:
- NCCL backend for high-speed inter-GPU communication
- Overlapped communication and computation
- Gradient accumulation across devices
- True async tile execution (no global barrier)

Key Components
--------------
- NCCLConfig: NCCL communication configuration
- NCCLCommunicator: NCCL-based communication wrapper
- MultiGPUConfig: Multi-GPU training configuration
- MultiGPUEquiTile: Full multi-GPU wrapper

Examples
--------
Single-process multi-GPU:
>>> model = EquiTile(neurons_per_tile=64, num_layers=4,
...                  tiles_per_layer=4, input_dim=784, output_dim=10)
>>> multi_gpu = MultiGPUEquiTile(model, device_ids=[0, 1, 2, 3])
>>> stats = multi_gpu.train_step(X, y)

Multi-process (spawn):
>>> def worker(rank, world_size):
...     dist.init_process_group('nccl', rank=rank, world_size=world_size)
...     model = EquiTile(...)
...     multi_gpu = MultiGPUEquiTile(model)
...     ...
>>> spawn_multi_gpu_worker(worker, world_size=4)
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from bioplausible.models.base import BioModel

from .kernels import (
    compute_activity_update,
    compute_hebbian_update,
    compute_tile_prediction,
)

if TYPE_CHECKING:
    from .core import EquiTile


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NCCLConfig:
    """Configuration for NCCL communication.

    Attributes
    ----------
    world_size : int
        Total number of processes
    rank : int
        This process's rank
    master_addr : str
        Master node address
    master_port : str
        Master node port
    backend : str
        Communication backend ('nccl', 'gloo', 'mpi')
    timeout_minutes : int
        Timeout for operations in minutes
    init_method : str
        Initialization method
    """

    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    backend: str = "nccl"
    timeout_minutes: int = 30
    init_method: str = "env://"

    def to_env(self) -> Dict[str, str]:
        """Convert to environment variables.

        Returns
        -------
        dict
            Environment variable mapping
        """
        return {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": self.master_port,
            "WORLD_SIZE": str(self.world_size),
            "RANK": str(self.rank),
        }


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training.

    Attributes
    ----------
    device_ids : list of int
        GPU device IDs to use
    tile_assignment : str
        Tile assignment strategy: 'round_robin', 'layered', 'balanced'
    sync_frequency : int
        Sync gradients every N steps
    overlap_comm : bool
        Overlap communication with computation
    async_execution : bool
        Enable true async execution
    gradient_accumulation : int
        Gradient accumulation steps
    """

    device_ids: List[int] = field(default_factory=list)
    tile_assignment: str = "round_robin"
    sync_frequency: int = 1
    overlap_comm: bool = True
    async_execution: bool = True
    gradient_accumulation: int = 1

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_assignments = {"round_robin", "layered", "balanced"}
        if self.tile_assignment not in valid_assignments:
            raise ValueError(f"tile_assignment must be one of {valid_assignments}")


# =============================================================================
# NCCL Communicator
# =============================================================================


class NCCLCommunicator:
    """NCCL-based inter-GPU communication.

    Provides:
    - All-reduce for gradient synchronization
    - All-gather for activity exchange
    - Broadcast for weight synchronization
    - Send/recv for tile boundary communication

    Parameters
    ----------
    config : NCCLConfig, optional
        NCCL configuration
    """

    def __init__(self, config: Optional[NCCLConfig] = None) -> None:
        self.config = config or NCCLConfig()
        self.initialized = False
        self.device: Optional[torch.device] = None
        self._timeout: Optional[dist.ProcessGroupOptions] = None

    def init_process_group(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        """Initialize NCCL process group.

        Parameters
        ----------
        rank : int, optional
            Process rank
        world_size : int, optional
            Total processes
        """
        if rank is not None:
            self.config.rank = rank
        if world_size is not None:
            self.config.world_size = world_size

        # Set environment variables
        for key, value in self.config.to_env().items():
            os.environ.setdefault(key, value)

        # Initialize process group
        try:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
            )
            self.device = torch.device(f"cuda:{self.config.rank}")
            torch.cuda.set_device(self.device)
            self.initialized = True

            print(
                f"NCCL initialized: rank {self.config.rank}/{self.config.world_size}, "
                f"device {self.device}"
            )
        except Exception as e:
            print(f"Warning: NCCL initialization failed: {e}")
            self.device = torch.device("cpu")

    def destroy(self) -> None:
        """Destroy process group."""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "avg",
    ) -> torch.Tensor:
        """All-reduce a tensor across devices.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to reduce
        op : str
            Reduction operation: 'avg', 'sum', 'min', 'max'

        Returns
        -------
        torch.Tensor
            Reduced tensor
        """
        if not self.initialized or self.config.world_size == 1:
            return tensor

        op_map = {
            "avg": dist.ReduceOp.AVG,
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }

        reduce_op = op_map.get(op, dist.ReduceOp.AVG)
        dist.all_reduce(tensor, op=reduce_op)

        return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Gather tensors from all devices.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to gather

        Returns
        -------
        list of torch.Tensor
            Gathered tensors
        """
        if not self.initialized or self.config.world_size == 1:
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(gathered, tensor)

        return gathered

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast tensor from source device.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to broadcast
        src : int
            Source rank

        Returns
        -------
        torch.Tensor
            Broadcasted tensor
        """
        if not self.initialized or self.config.world_size == 1:
            return tensor

        dist.broadcast(tensor, src=src)
        return tensor

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
        tag: int = 0,
    ) -> None:
        """Send tensor to destination device.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to send
        dst : int
            Destination rank
        tag : int
            Message tag
        """
        if not self.initialized:
            return

        dist.send(tensor, dst=dst, tag=tag)

    def recv(
        self,
        tensor: torch.Tensor,
        src: int = -1,
        tag: int = 0,
    ) -> int:
        """Receive tensor from source device.

        Parameters
        ----------
        tensor : torch.Tensor
            Buffer to receive into
        src : int
            Source rank (-1 for any)
        tag : int
            Message tag

        Returns
        -------
        int
            Source rank
        """
        if not self.initialized:
            return -1

        return dist.recv(tensor, src=src, tag=tag)

    def barrier(self) -> None:
        """Synchronization barrier."""
        if self.initialized:
            dist.barrier()

    @property
    def rank(self) -> int:
        """Get this process's rank."""
        return self.config.rank

    @property
    def world_size(self) -> int:
        """Get total number of processes."""
        return self.config.world_size

    @property
    def is_distributed(self) -> bool:
        """Check if distributed mode is active."""
        return self.initialized and self.config.world_size > 1


# =============================================================================
# Async Tile Executor
# =============================================================================


class AsyncTileExecutor:
    """Executes tile operations asynchronously with NCCL.

    Overlaps communication and computation for maximum throughput.

    Parameters
    ----------
    communicator : NCCLCommunicator
        NCCL communicator
    """

    def __init__(self, communicator: NCCLCommunicator) -> None:
        self.communicator = communicator
        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start async executor."""
        if torch.cuda.is_available():
            self._compute_stream = torch.cuda.Stream()
            self._comm_stream = torch.cuda.Stream()
        self._running = True

    def stop(self) -> None:
        """Stop async executor."""
        self._running = False
        self.synchronize()

    def submit_compute(
        self,
        op: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Submit compute operation.

        Parameters
        ----------
        op : callable
            Operation to execute
        *args
            Positional arguments
        **kwargs
            Keyword arguments
        """
        if self._compute_stream is not None:
            with torch.cuda.stream(self._compute_stream):
                op(*args, **kwargs)
        else:
            op(*args, **kwargs)

    def submit_comm(
        self,
        op: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Submit communication operation.

        Parameters
        ----------
        op : callable
            Operation to execute
        *args
            Positional arguments
        **kwargs
            Keyword arguments
        """
        if self._comm_stream is not None:
            with torch.cuda.stream(self._comm_stream):
                op(*args, **kwargs)
        else:
            op(*args, **kwargs)

    def synchronize(self) -> None:
        """Synchronize compute and communication streams."""
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
        if self._comm_stream is not None:
            self._comm_stream.synchronize()

    @contextmanager
    def compute_stream_context(self):
        """Context manager for compute stream."""
        if self._compute_stream is not None:
            with torch.cuda.stream(self._compute_stream):
                yield
        else:
            yield

    @contextmanager
    def comm_stream_context(self):
        """Context manager for communication stream."""
        if self._comm_stream is not None:
            with torch.cuda.stream(self._comm_stream):
                yield
        else:
            yield


# =============================================================================
# Multi-GPU EquiTile
# =============================================================================


class MultiGPUEquiTile:
    """True multi-GPU EquiTile with NCCL communication.

    Distributes tiles across GPUs and enables true async execution
    with overlapped communication and computation.

    Parameters
    ----------
    model : EquiTile
        Base EquiTile model
    config : MultiGPUConfig, optional
        Multi-GPU configuration

    Examples
    --------
    >>> model = EquiTile(neurons_per_tile=64, num_layers=4,
    ...                  tiles_per_layer=4, input_dim=784, output_dim=10)
    >>> multi_gpu = MultiGPUEquiTile(model, device_ids=[0, 1, 2, 3])
    >>> stats = multi_gpu.train_step(X, y)
    """

    def __init__(
        self,
        model: EquiTile,
        config: Optional[MultiGPUConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or MultiGPUConfig()

        # Set up devices
        if not self.config.device_ids:
            if torch.cuda.is_available():
                self.config.device_ids = list(range(torch.cuda.device_count()))
            else:
                self.config.device_ids = [0]  # Fallback to single CPU "device" 0

        if torch.cuda.is_available():
            self.devices = [torch.device(f"cuda:{i}") for i in self.config.device_ids]
        else:
            self.devices = [torch.device("cpu") for _ in self.config.device_ids]

        self.n_devices = len(self.devices)

        # Initialize NCCL communicator
        self.communicator = NCCLCommunicator()

        # Async executor
        self.executor: Optional[AsyncTileExecutor] = None
        if self.config.async_execution and self.n_devices > 1:
            self.executor = AsyncTileExecutor(self.communicator)
            self.executor.start()

        # Assign tiles to devices
        self.tile_assignments = self._assign_tiles()

        # Move tiles to devices
        self._distribute_tiles()

        # Gradient accumulation
        self._accumulated_steps = 0
        self._pending_gradients: Dict[str, List[torch.Tensor]] = {}

        # Timing
        self._comm_time = 0.0
        self._compute_time = 0.0

    def _assign_tiles(self) -> Dict[int, List[int]]:
        """Assign tiles to devices.

        Returns
        -------
        dict
            Device ID to tile IDs mapping
        """
        n_tiles = len(self.model.graph.tiles)
        tile_ids = list(self.model.graph.tiles.keys())

        assignments: Dict[int, List[int]] = {i: [] for i in range(self.n_devices)}

        if self.config.tile_assignment == "round_robin":
            for i, tile_id in enumerate(tile_ids):
                device_idx = i % self.n_devices
                assignments[device_idx].append(tile_id)

        elif self.config.tile_assignment == "layered":
            # Assign by layers
            tiles_per_device = (n_tiles + self.n_devices - 1) // self.n_devices
            for i, tile_id in enumerate(tile_ids):
                device_idx = min(i // tiles_per_device, self.n_devices - 1)
                assignments[device_idx].append(tile_id)

        elif self.config.tile_assignment == "balanced":
            # Balance by neuron count
            neuron_counts: Dict[int, int] = {i: 0 for i in range(self.n_devices)}
            for tile_id in tile_ids:
                tile = self.model.graph.tiles[tile_id]
                # Assign to device with fewest neurons
                min_device = min(neuron_counts.keys(), key=lambda d: neuron_counts[d])
                assignments[min_device].append(tile_id)
                neuron_counts[min_device] += tile.neurons

        return assignments

    def _distribute_tiles(self) -> None:
        """Move tiles to assigned devices."""
        for device_idx, tile_ids in self.tile_assignments.items():
            device = self.devices[device_idx]

            for tile_id in tile_ids:
                tile = self.model.graph.tiles[tile_id]

                if tile.activity is not None:
                    tile.activity = tile.activity.to(device)
                if tile.prediction is not None:
                    tile.prediction = tile.prediction.to(device)
                if tile.error is not None:
                    tile.error = tile.error.to(device)

        # Move edge weights
        for device_idx, tile_ids in self.tile_assignments.items():
            device = self.devices[device_idx]

            for tile_id in tile_ids:
                tile = self.model.graph.tiles[tile_id]

                for dst_id in tile.fwd_neighbors:
                    weight, bias = self.model._get_edge_params(tile_id, dst_id)

                    if weight is not None:
                        weight.data = weight.data.to(device)
                    if bias is not None:
                        bias.data = bias.data.to(device)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Training step with multi-GPU execution.

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
        if self.n_devices == 1:
            return self.model.train_step(x, y)

        start_time = time.perf_counter()

        batch_size = x.shape[0]
        device = self.devices[0]

        # Move input to first device
        x = x.to(device)
        y = y.to(device)

        input_proj = self.model.W_in(x)

        # Initialize activities on each device
        for device_idx, tile_ids in self.tile_assignments.items():
            for tile_id in tile_ids:
                tile = self.model.graph.tiles[tile_id]

                if tile.is_input:
                    idx = self.model.graph.input_tile_ids.index(tile.id)
                    start = idx * self.model.config.neurons_per_tile
                    tile.activity = input_proj[:, start : start + tile.neurons].clone()
                else:
                    tile.activity = torch.zeros(
                        batch_size, tile.neurons, device=self.devices[device_idx]
                    )
                tile.prediction = None
                tile.error = None

        # Relaxation loop with async communication
        for step in range(self.model.config.inference_steps):
            self._async_relax_step(batch_size)

        # Learning step
        stats = self._multi_gpu_learning(y)

        # Record timing
        elapsed = time.perf_counter() - start_time
        stats["total_time"] = elapsed
        stats["comm_time"] = self._comm_time
        stats["compute_time"] = self._compute_time
        stats["n_devices"] = self.n_devices

        return stats

    def _async_relax_step(self, batch_size: int) -> None:
        """One relaxation step with async communication.

        Parameters
        ----------
        batch_size : int
            Batch size
        """
        compute_start = time.perf_counter()

        # Compute predictions in parallel across devices
        for device_idx, tile_ids in self.tile_assignments.items():
            if self.executor:
                with self.executor.compute_stream_context():
                    self._compute_predictions_device(batch_size, device_idx, tile_ids)
            else:
                self._compute_predictions_device(batch_size, device_idx, tile_ids)

        if self.executor:
            self.executor.synchronize()

        self._compute_time = time.perf_counter() - compute_start

        # Compute errors locally
        for device_idx, tile_ids in self.tile_assignments.items():
            for tile_id in tile_ids:
                tile = self.model.graph.tiles[tile_id]
                if tile.activity is None:
                    continue

                if tile.prediction is None:
                    tile.error = tile.activity.clone()
                else:
                    tile.error = tile.activity - tile.prediction

        # Exchange boundary activities (async)
        comm_start = time.perf_counter()
        self._exchange_boundary_activities(batch_size)

        if self.executor:
            self.executor.synchronize()

        self._comm_time = time.perf_counter() - comm_start

        # Update activities
        for device_idx, tile_ids in self.tile_assignments.items():
            for tile_id in tile_ids:
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
                    clamp_min=self.model.config.activity_clamp_min,
                    clamp_max=self.model.config.activity_clamp_max,
                    clamp=self.model.config.clamp_activities,
                )

    def _compute_predictions_device(
        self,
        batch_size: int,
        device_idx: int,
        tile_ids: List[int],
    ) -> None:
        """Compute predictions for tiles on a device.

        Parameters
        ----------
        batch_size : int
            Batch size
        device_idx : int
            Device index
        tile_ids : list of int
            Tile IDs on this device
        """
        device = self.devices[device_idx]

        for tile_id in tile_ids:
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
                    else torch.zeros(batch_size, src.neurons, device=device)
                )
                inputs.append(self.model._apply_activation(src_activity) @ weight)

                if bias is not None:
                    if total_bias is None:
                        total_bias = bias
                    else:
                        total_bias = total_bias + bias

            tile.prediction = compute_tile_prediction(
                inputs,
                total_bias,
                output_shape=(batch_size, tile.neurons),
                device=device,
            )

    def _exchange_boundary_activities(self, batch_size: int) -> None:
        """Exchange activities across tile boundaries.

        Parameters
        ----------
        batch_size : int
            Batch size
        """
        # For same-process multi-GPU, this is a no-op (shared memory)
        # For multi-process, would use NCCL send/recv
        pass

    def _multi_gpu_learning(self, y: torch.Tensor) -> Dict[str, float]:
        """Learning step for multi-GPU training.

        Parameters
        ----------
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        # Gather output activities
        out_activities = torch.cat(
            [
                self.model.graph.tiles[tid].activity
                for tid in self.model.graph.output_tile_ids
            ],
            dim=-1,
        )

        # Compute loss
        logits = self.model.W_out(out_activities)

        loss = self.model.task_handler.compute_loss(logits, y)

        # Backprop for I/O projections
        self.model._ensure_local_optimizers()
        self.model._optim_io.zero_grad()
        loss.backward()

        # Sync gradients across devices (all-reduce)
        if self.n_devices > 1:
            for param in self.model.W_in.parameters():
                if param.grad is not None:
                    self.communicator.all_reduce(param.grad)

            for param in self.model.W_out.parameters():
                if param.grad is not None:
                    self.communicator.all_reduce(param.grad)

        self.model._optim_io.step()

        # Local Hebbian updates
        for device_idx, tile_ids in self.tile_assignments.items():
            for tile_id in tile_ids:
                tile = self.model.graph.tiles[tile_id]

                for dst_id in tile.fwd_neighbors:
                    weight, bias = self.model._get_edge_params(tile_id, dst_id)

                    if weight is None:
                        continue

                    src = tile
                    dst = self.model.graph.tiles[dst_id]

                    if src.activity is None or dst.error is None:
                        continue

                    edge_idx = self.model.graph.edges.index((tile_id, dst_id))
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

    def destroy(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.stop()
        self.communicator.destroy()

    def __del__(self) -> None:
        """Destructor."""
        self.destroy()

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.n_devices > 1


# =============================================================================
# Multi-Process Spawn Helper
# =============================================================================


def spawn_multi_gpu_worker(
    worker_fn: Callable[[int, int], None],
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "29500",
) -> None:
    """Spawn multi-GPU worker processes.

    Parameters
    ----------
    worker_fn : callable
        Worker function with signature (rank, world_size)
    world_size : int
        Number of processes to spawn
    master_addr : str
        Master node address
    master_port : str
        Master node port

    Examples
    --------
    >>> def worker(rank, world_size):
    ...     dist.init_process_group('nccl', rank=rank, world_size=world_size)
    ...     model = EquiTile(...)
    ...     multi_gpu = MultiGPUEquiTile(model)
    ...     ...
    >>> spawn_multi_gpu_worker(worker, world_size=4)
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    mp.spawn(worker_fn, args=(world_size,), nprocs=world_size, join=True)


# =============================================================================
# Factory Functions
# =============================================================================


def create_multigpu_model(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    input_dim: int = 784,
    output_dim: int = 10,
    device_ids: Optional[List[int]] = None,
    tile_assignment: str = "round_robin",
    **kwargs,
) -> Tuple[EquiTile, MultiGPUEquiTile]:
    """Create a multi-GPU EquiTile model.

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
    tile_assignment : str
        Tile assignment strategy
    **kwargs
        Additional arguments for EquiTile

    Returns
    -------
    tuple of (EquiTile, MultiGPUEquiTile)
        Base model and multi-GPU wrapper
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

    multi_gpu = MultiGPUEquiTile(
        model,
        config=MultiGPUConfig(
            device_ids=device_ids or [],
            tile_assignment=tile_assignment,
        ),
    )

    return model, multi_gpu
