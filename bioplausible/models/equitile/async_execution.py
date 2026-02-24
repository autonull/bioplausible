"""
EquiTile Async: Asynchronous Tile Execution
============================================

Enables parallel, asynchronous tile processing for:
- Multi-GPU distributed training
- CPU/GPU heterogeneous execution
- True async updates (no global synchronization)

Key Components
--------------
- TileTask: Task descriptor for tile processing
- TileResult: Result from tile processing
- TileProcessor: Processes individual tiles
- TileScheduler: Manages tile execution queue
- AsyncEquiTile: Async-enabled EquiTile wrapper

Examples
--------
>>> from bioplausible.models.equitile import EquiTile, AsyncEquiTile, AsyncConfig
>>> model = EquiTile(neurons_per_tile=64, num_layers=4, tiles_per_layer=4,
...                  input_dim=784, output_dim=10)
>>> async_model = AsyncEquiTile(model, config=AsyncConfig(n_workers=4))
>>> with async_model.async_context():
...     stats = async_model.train_step(X, y)
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from bioplausible.models.base import BioModel
from .kernels import (
    compute_tile_prediction,
    compute_activity_update,
    compute_hebbian_update,
)

if TYPE_CHECKING:
    from .core import EquiTile
    from .topology import TileState


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(order=True)
class TileTask:
    """A task for processing a single tile.

    Attributes
    ----------
    tile_id : int
        Tile identifier
    phase : str
        Processing phase: 'predict', 'update', or 'learn'
    input_data : Optional[Dict[str, Any]]
        Additional input data for the task
    priority : float
        Task priority (higher = more urgent)
    created_at : float
        Timestamp when task was created
    """
    priority: float
    tile_id: int = field(compare=False)
    phase: str = field(compare=False)
    input_data: Optional[Dict[str, Any]] = field(default_factory=dict, compare=False)
    created_at: float = field(default_factory=time.perf_counter, compare=False)

    @classmethod
    def create(
        cls,
        tile_id: int,
        phase: str,
        priority: float = 0.0,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> TileTask:
        """Factory method for creating TileTask.

        Parameters
        ----------
        tile_id : int
            Tile identifier
        phase : str
            Processing phase
        priority : float
            Task priority
        input_data : dict, optional
            Additional input data

        Returns
        -------
        TileTask
            New task instance
        """
        return cls(
            tile_id=tile_id,
            phase=phase,
            priority=-priority,  # Negate for min-heap (higher priority = lower value)
            input_data=input_data or {},
        )


@dataclass
class TileResult:
    """Result from processing a tile.

    Attributes
    ----------
    tile_id : int
        Tile identifier
    phase : str
        Processing phase
    success : bool
        Whether processing succeeded
    data : Optional[Dict[str, Any]]
        Result data if successful
    error : Optional[Exception]
        Exception if failed
    elapsed_time : float
        Processing time in seconds
    """
    tile_id: int
    phase: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    elapsed_time: float = 0.0

    @property
    def is_failed(self) -> bool:
        """Check if result represents a failure."""
        return not self.success


# =============================================================================
# Tile Processor
# =============================================================================

class TileProcessor:
    """Processes individual tiles.

    Can run on CPU, GPU, or other devices.

    Parameters
    ----------
    device : torch.device
        Device to run processing on
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device('cpu')

    def process(
        self,
        model: EquiTile,
        task: TileTask,
    ) -> TileResult:
        """Process a single tile task.

        Parameters
        ----------
        model : EquiTile
            The model containing tile state
        task : TileTask
            Task to execute

        Returns
        -------
        TileResult
            Processing result
        """
        start_time = time.perf_counter()

        try:
            if task.tile_id not in model.graph.tiles:
                raise KeyError(f"Tile {task.tile_id} not found in model graph")

            tile = model.graph.tiles[task.tile_id]

            if task.phase == 'predict':
                result = self._compute_prediction(model, tile, task.input_data or {})
            elif task.phase == 'update':
                result = self._update_activity(model, tile, task.input_data or {})
            elif task.phase == 'learn':
                result = self._compute_weight_update(model, tile, task.input_data or {})
            else:
                raise ValueError(f"Unknown phase: {task.phase}. Must be 'predict', 'update', or 'learn'")

            elapsed = time.perf_counter() - start_time

            return TileResult(
                tile_id=task.tile_id,
                phase=task.phase,
                success=True,
                data=result,
                elapsed_time=elapsed,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return TileResult(
                tile_id=task.tile_id,
                phase=task.phase,
                success=False,
                error=e,
                elapsed_time=elapsed,
            )

    def _compute_prediction(
        self,
        model: EquiTile,
        tile: TileState,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute prediction for a tile.

        Parameters
        ----------
        model : EquiTile
            The model
        tile : TileState
            Tile state
        input_data : dict
            Additional input data

        Returns
        -------
        dict
            Prediction result
        """
        batch_size = input_data.get('batch_size', 1)
        device = input_data.get('device', self.device)

        if tile.is_input:
            return {'prediction': None}

        inputs = []
        total_bias = None

        for src_id in tile.bwd_neighbors:
            src = model.graph.tiles[src_id]
            weight, bias = model._get_edge_params(src_id, tile.id)

            if weight is None:
                continue

            src_activity = (
                src.activity
                if src.activity is not None
                else torch.zeros(batch_size, src.neurons, device=device)
            )
            inputs.append(model._apply_activation(src_activity) @ weight)

            if bias is not None:
                if total_bias is None:
                    total_bias = bias
                else:
                    total_bias = total_bias + bias

        pred = compute_tile_prediction(
            inputs,
            total_bias,
            output_shape=(batch_size, tile.neurons),
            device=device
        )

        tile.prediction = pred
        return {'prediction': pred}

    def _update_activity(
        self,
        model: EquiTile,
        tile: TileState,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update tile activity.

        Parameters
        ----------
        model : EquiTile
            The model
        tile : TileState
            Tile state
        input_data : dict
            Additional input data

        Returns
        -------
        dict
            Updated activity
        """
        if tile.is_input:
            return {'activity': tile.activity}

        if tile.error is None:
            return {'activity': tile.activity}

        # Get importance
        tile_idx = list(model.graph.tiles.keys()).index(tile.id)
        imp = torch.sigmoid(model.tile_importance[tile_idx]).item()

        fwd_feedback = []
        for dst_id in tile.fwd_neighbors:
            dst = model.graph.tiles[dst_id]
            weight, _ = model._get_edge_params(tile.id, dst_id)
            if weight is not None and dst.error is not None:
                fwd_feedback.append(dst.error @ weight.T)

        tile.activity = compute_activity_update(
            activity=tile.activity,
            error=tile.error,
            fwd_feedback=fwd_feedback,
            importance=imp,
            step_size=model.config.step_size,
            lambda_error=model.config.lambda_error,
            clamp_min=model.config.activity_clamp_min,
            clamp_max=model.config.activity_clamp_max,
            clamp=model.config.clamp_activities,
        )

        return {'activity': tile.activity}

    def _compute_weight_update(
        self,
        model: EquiTile,
        tile: TileState,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute weight updates for edges connected to this tile.

        Parameters
        ----------
        model : EquiTile
            The model
        tile : TileState
            Tile state
        input_data : dict
            Additional input data

        Returns
        -------
        dict
            Weight updates
        """
        updates: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        for dst_id in tile.fwd_neighbors:
            edge_key = (tile.id, dst_id)
            if edge_key not in model.graph.edges:
                continue

            weight, _ = model._get_edge_params(*edge_key)
            if weight is None:
                continue

            dst = model.graph.tiles[dst_id]

            if tile.activity is None or dst.error is None:
                continue

            edge_idx = model.graph.edges.index(edge_key)
            imp = torch.sigmoid(model.edge_importance[edge_idx]).item()

            src_act = model._apply_activation(tile.activity)
            dst_err = dst.error

            batch_size = input_data.get('batch_size', src_act.shape[0])

            weight_update, bias_update = compute_hebbian_update(
                src_act, dst_err, imp, batch_size
            )

            updates[edge_key] = (weight_update, bias_update)

        return {'updates': updates}


# =============================================================================
# Tile Scheduler
# =============================================================================

class TileScheduler:
    """Schedules tile tasks for async execution.

    Manages priority queue and distributes work across processors.

    Parameters
    ----------
    n_workers : int
        Number of worker threads/processes
    use_processes : bool
        If True, use ProcessPoolExecutor instead of ThreadPoolExecutor
    """

    def __init__(
        self,
        n_workers: int = 4,
        use_processes: bool = False,
    ) -> None:
        self.n_workers = max(1, n_workers)
        self.use_processes = use_processes

        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self._running = False
        self._futures: List[Future] = []

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        self._running = True
        self._futures.clear()

    def stop(self, wait: bool = True) -> None:
        """Stop the scheduler.

        Parameters
        ----------
        wait : bool
            If True, wait for pending tasks to complete
        """
        self._running = False

        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

        self._futures.clear()

    def submit(self, task: TileTask) -> None:
        """Submit a tile task.

        Parameters
        ----------
        task : TileTask
            Task to submit
        """
        if not self._running:
            raise RuntimeError("Scheduler not started. Call start() first.")
        self._task_queue.put(task)

    def process_batch(
        self,
        model: EquiTile,
        processor: TileProcessor,
        timeout: Optional[float] = None,
    ) -> List[TileResult]:
        """Process all pending tasks in batch.

        Parameters
        ----------
        model : EquiTile
            The model
        processor : TileProcessor
            Tile processor
        timeout : float, optional
            Timeout for each task in seconds

        Returns
        -------
        list of TileResult
            Processing results
        """
        if not self._running or self._executor is None:
            return []

        # Collect pending tasks
        tasks: List[TileTask] = []
        while not self._task_queue.empty():
            task = self._task_queue.get()
            tasks.append(task)

        if not tasks:
            return []

        # Submit all tasks
        self._futures.clear()
        for task in tasks:
            future = self._executor.submit(processor.process, model, task)
            self._futures.append(future)

        # Collect results
        results: List[TileResult] = []
        for future in self._futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except TimeoutError as e:
                results.append(TileResult(
                    tile_id=0,
                    phase='unknown',
                    success=False,
                    error=e,
                ))
            except Exception as e:
                results.append(TileResult(
                    tile_id=0,
                    phase='unknown',
                    success=False,
                    error=e,
                ))

        return results

    @property
    def pending_tasks(self) -> int:
        """Number of pending tasks in queue."""
        return self._task_queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AsyncConfig:
    """Configuration for async execution.

    Attributes
    ----------
    n_workers : int
        Number of worker threads/processes
    use_processes : bool
        If True, use processes instead of threads
    device_ids : list of int
        GPU device IDs to use
    batch_threshold : int
        Minimum batch size to enable async execution
    priority_alpha : float
        Weight for error in priority calculation
    priority_beta : float
        Weight for importance in priority calculation
    """
    n_workers: int = 4
    use_processes: bool = False
    device_ids: List[int] = field(default_factory=list)
    batch_threshold: int = 32
    priority_alpha: float = 0.5
    priority_beta: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        if not 0 <= self.priority_alpha <= 1:
            raise ValueError("priority_alpha must be in [0, 1]")
        if not 0 <= self.priority_beta <= 1:
            raise ValueError("priority_beta must be in [0, 1]")


# =============================================================================
# Async EquiTile Wrapper
# =============================================================================

class AsyncEquiTile:
    """Async-enabled EquiTile wrapper.

    Wraps an EquiTile model to enable asynchronous tile execution.

    Parameters
    ----------
    model : EquiTile
        Base EquiTile model
    config : AsyncConfig, optional
        Async execution configuration

    Examples
    --------
    >>> model = EquiTile(neurons_per_tile=64, num_layers=4,
    ...                  tiles_per_layer=4, input_dim=784, output_dim=10)
    >>> async_model = AsyncEquiTile(model, config=AsyncConfig(n_workers=4))
    >>> with async_model.async_context():
    ...     stats = async_model.train_step(X, y)
    """

    def __init__(
        self,
        model: EquiTile,
        config: Optional[AsyncConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or AsyncConfig()

        self._scheduler: Optional[TileScheduler] = None
        self._processors: List[TileProcessor] = []
        self._async_context = False

        # Set up devices
        if self.config.device_ids:
            self.devices = [
                torch.device(f'cuda:{i}') for i in self.config.device_ids
            ]
        else:
            self.devices = [torch.device('cpu')]

    @contextmanager
    def async_context(self):
        """Context manager for async execution.

        Yields
        ------
        AsyncEquiTile
            Self for use in context
        """
        self._async_context = True
        self._scheduler = TileScheduler(
            n_workers=self.config.n_workers,
            use_processes=self.config.use_processes,
        )
        self._scheduler.start()
        self._processors = [
            TileProcessor(device) for device in self.devices
        ]
        try:
            yield self
        finally:
            self._scheduler.stop(wait=True)
            self._scheduler = None
            self._processors = []
            self._async_context = False

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Training step with optional async execution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, input_dim)
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        if not self._async_context:
            # Fall back to sync execution
            return self.model.train_step(x, y)

        batch_size = x.shape[0]

        # For small batches, use sync execution
        if batch_size < self.config.batch_threshold:
            return self.model.train_step(x, y)

        # Async execution
        return self._async_train_step(x, y)

    def _async_train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Async training step.

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
        device = x.device
        batch_size = x.shape[0]

        input_proj = self.model.W_in(x)

        # Initialize activities using core API
        self.model._init_activities(input_proj, batch_size, device)

        # Async relaxation loop
        for _ in range(self.model.config.inference_steps):
            self._async_relax_step(input_proj)

        # Task-driven learning (sync for now)
        return self._sync_learning(x, y)

    def _async_relax_step(self, input_proj: torch.Tensor) -> None:
        """One async relaxation step.

        Parameters
        ----------
        input_proj : torch.Tensor
            Projected input
        """
        batch_size = input_proj.shape[0]
        device = input_proj.device

        # Compute priorities for all tiles
        priorities = self._compute_tile_priorities()

        if self._scheduler is None or not self._processors:
            return

        # Submit prediction tasks
        for tile in self.model.graph.all_tiles:
            if not tile.is_input:
                task = TileTask.create(
                    tile_id=tile.id,
                    phase='predict',
                    priority=priorities.get(tile.id, 0.0),
                    input_data={'batch_size': batch_size, 'device': device},
                )
                self._scheduler.submit(task)

        # Process predictions
        processor = self._processors[0]
        results = self._scheduler.process_batch(self.model, processor)

        # Handle errors
        for result in results:
            if result.is_failed and result.error is not None:
                raise RuntimeError(f"Prediction failed for tile {result.tile_id}: {result.error}")

        # Compute errors (sync, fast)
        self.model._compute_errors()

        # Submit update tasks
        for tile in self.model.graph.all_tiles:
            if not tile.is_input and tile.error is not None:
                task = TileTask.create(
                    tile_id=tile.id,
                    phase='update',
                    priority=priorities.get(tile.id, 0.0),
                    input_data={},
                )
                self._scheduler.submit(task)

        # Process updates
        results = self._scheduler.process_batch(self.model, processor)

        # Handle errors
        for result in results:
            if result.is_failed and result.error is not None:
                raise RuntimeError(f"Update failed for tile {result.tile_id}: {result.error}")

    def _sync_learning(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Synchronous learning step (local Hebbian updates).

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
        return self.model._pc_learning(x, y, batch_size)

    def _compute_tile_priorities(self) -> Dict[int, float]:
        """Compute priority scores for all tiles.

        Returns
        -------
        dict
            Priority scores per tile
        """
        priorities: Dict[int, float] = {}

        for i, tile in enumerate(self.model.graph.all_tiles):
            if tile.is_input:
                priorities[tile.id] = 0.0
                continue

            # Error magnitude
            error_mag = 0.0
            if tile.error is not None:
                error_mag = tile.error.norm(p=2, dim=-1).mean().item()

            # Importance
            importance = torch.sigmoid(self.model.tile_importance[i]).item()

            # Priority = alpha * error + beta * importance
            priority = (
                self.config.priority_alpha * error_mag +
                self.config.priority_beta * importance
            )
            priorities[tile.id] = priority

        return priorities

    @property
    def is_async(self) -> bool:
        """Check if currently in async context."""
        return self._async_context

    @property
    def scheduler(self) -> Optional[TileScheduler]:
        """Get the scheduler (only available in async context)."""
        return self._scheduler


# =============================================================================
# Factory Functions
# =============================================================================

def create_async_model(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    input_dim: int = 784,
    output_dim: int = 10,
    n_workers: int = 4,
    use_processes: bool = False,
    **kwargs,
) -> Tuple[EquiTile, AsyncEquiTile]:
    """Create an async-enabled EquiTile model.

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
    n_workers : int
        Number of async workers
    use_processes : bool
        If True, use processes instead of threads
    **kwargs
        Additional arguments for EquiTile

    Returns
    -------
    tuple of (EquiTile, AsyncEquiTile)
        Base model and async wrapper
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

    async_model = AsyncEquiTile(
        model,
        config=AsyncConfig(
            n_workers=n_workers,
            use_processes=use_processes,
        ),
    )

    return model, async_model
