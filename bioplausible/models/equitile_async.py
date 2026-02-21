"""
EquiTile Async: Asynchronous Tile Execution
============================================

Enables parallel, asynchronous tile processing for:
- Multi-GPU distributed training
- CPU/GPU heterogeneous execution
- True async updates (no global synchronization)

Key Components
--------------
- TileProcessor: Processes individual tiles
- TileScheduler: Manages tile execution queue
- AsyncEquiTile: Async-enabled EquiTile wrapper
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .equitile import EquiTile, TileState


@dataclass
class TileTask:
    """A task for processing a single tile."""
    tile_id: int
    phase: str  # 'predict', 'update', 'learn'
    input_data: Optional[Dict[str, Any]] = None
    priority: float = 0.0
    future: Optional[torch.Future] = None

    def __lt__(self, other):
        """Enable comparison for priority queue."""
        return self.priority > other.priority  # Higher priority = lower in queue


@dataclass
class TileResult:
    """Result from processing a tile."""
    tile_id: int
    phase: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class TileProcessor:
    """Processes individual tiles.

    Can run on CPU, GPU, or other devices.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')

    def process(self, model: 'EquiTile', task: TileTask) -> TileResult:
        """Process a single tile task."""
        try:
            tile = model.graph.tiles[task.tile_id]

            if task.phase == 'predict':
                result = self._compute_prediction(model, tile, task.input_data)
            elif task.phase == 'update':
                result = self._update_activity(model, tile, task.input_data)
            elif task.phase == 'learn':
                result = self._compute_weight_update(model, tile, task.input_data)
            else:
                raise ValueError(f"Unknown phase: {task.phase}")

            return TileResult(
                tile_id=task.tile_id,
                phase=task.phase,
                success=True,
                data=result
            )
        except Exception as e:
            return TileResult(
                tile_id=task.tile_id,
                phase=task.phase,
                success=False,
                error=e
            )

    def _compute_prediction(self, model: 'EquiTile', tile: 'TileState',
                           input_data: Dict) -> Dict:
        """Compute prediction for a tile."""
        batch_size = input_data.get('batch_size', 1)
        device = input_data.get('device', self.device)

        if tile.is_input:
            return {'prediction': None}

        pred = torch.zeros(batch_size, tile.neurons, device=device)

        for src_id in tile.bwd_neighbors:
            src = model.graph.tiles[src_id]
            edge = model.graph.edges.get((src_id, tile.id))

            if edge is None or edge.weight is None:
                continue

            src_activity = src.activity if src.activity is not None else torch.zeros(
                batch_size, src.neurons, device=device
            )
            pred = pred + model._apply_activation(src_activity) @ edge.weight

        if edge and edge.bias is not None:
            pred = pred + edge.bias.unsqueeze(0)

        tile.prediction = pred
        return {'prediction': pred}

    def _update_activity(self, model: 'EquiTile', tile: 'TileState',
                        input_data: Dict) -> Dict:
        """Update tile activity."""
        if tile.is_input:
            return {'activity': tile.activity}

        if tile.error is None:
            return {'activity': tile.activity}

        # Get importance
        tile_idx = list(model.graph.tiles.keys()).index(tile.id)
        imp = torch.sigmoid(model.tile_importance[tile_idx]).item()

        # Compute gradient
        grad = tile.error + model.config.lambda_error * tile.activity

        # Top-down modulation
        for dst_id in tile.fwd_neighbors:
            dst = model.graph.tiles[dst_id]
            edge = model.graph.edges.get((tile.id, dst_id))
            if edge and edge.weight is not None and dst.error is not None:
                grad = grad + dst.error @ edge.weight.T

        # Update
        delta = model.config.step_size * imp * grad
        tile.activity = tile.activity - delta

        if model.config.clamp_activities:
            tile.activity = torch.clamp(tile.activity, -5.0, 5.0)

        return {'activity': tile.activity}

    def _compute_weight_update(self, model: 'EquiTile', tile: 'TileState',
                               input_data: Dict) -> Dict:
        """Compute weight updates for edges connected to this tile."""
        updates = {}

        for dst_id in tile.fwd_neighbors:
            edge_key = (tile.id, dst_id)
            if edge_key not in model.graph.edges:
                continue

            edge = model.graph.edges[edge_key]
            dst = model.graph.tiles[dst_id]

            if tile.activity is None or dst.error is None:
                continue

            edge_idx = list(model.graph.edges.keys()).index(edge_key)
            imp = torch.sigmoid(model.edge_importance[edge_idx]).item()

            src_act = model._apply_activation(tile.activity)
            dst_err = dst.error

            weight_update = imp * (src_act.T @ dst_err) / input_data.get('batch_size', 1)
            bias_update = imp * dst_err.mean(dim=0) / input_data.get('batch_size', 1)

            updates[edge_key] = (weight_update, bias_update)

        return {'updates': updates}


class TileScheduler:
    """Schedules tile tasks for async execution.

    Manages priority queue and distributes work across processors.
    """

    def __init__(self, n_workers: int = 4, use_processes: bool = False):
        self.n_workers = n_workers
        self.use_processes = use_processes

        self._task_queue = queue.PriorityQueue()
        self._result_queue = queue.Queue()
        self._executor = None
        self._running = False

    def start(self):
        """Start the scheduler."""
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        self._running = True

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=True)

    def submit(self, task: TileTask):
        """Submit a tile task."""
        self._task_queue.put((-task.priority, task))

    def process_batch(self, model: 'EquiTile', processor: TileProcessor,
                     timeout: float = None) -> List[TileResult]:
        """Process all pending tasks in batch."""
        tasks = []
        while not self._task_queue.empty():
            _, task = self._task_queue.get()
            tasks.append(task)

        if not tasks:
            return []

        # Submit all tasks
        futures = []
        for task in tasks:
            future = self._executor.submit(processor.process, model, task)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append(TileResult(
                    tile_id=0,
                    phase='unknown',
                    success=False,
                    error=e
                ))

        return results


@dataclass
class AsyncConfig:
    """Configuration for async execution."""
    n_workers: int = 4
    use_processes: bool = False
    device_ids: List[int] = field(default_factory=list)
    batch_threshold: int = 32  # Min batch size for async
    priority_alpha: float = 0.5  # Weight for error in priority
    priority_beta: float = 0.5   # Weight for importance in priority


class AsyncEquiTile:
    """Async-enabled EquiTile wrapper.

    Wraps an EquiTile model to enable asynchronous tile execution.

    Usage:
        model = EquiTile(...)
        async_model = AsyncEquiTile(model, n_workers=4)

        with async_model.async_context():
            stats = async_model.train_step(X, y)
    """

    def __init__(self, model: 'EquiTile', config: Optional[AsyncConfig] = None):
        self.model = model
        self.config = config or AsyncConfig()

        self._scheduler = None
        self._processors = []
        self._async_context = False

        # Set up devices
        if self.config.device_ids:
            self.devices = [
                torch.device(f'cuda:{i}') for i in self.config.device_ids
            ]
        else:
            self.devices = [torch.device('cpu')]

    def async_context(self):
        """Context manager for async execution."""
        return _AsyncContextManager(self)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Training step with optional async execution."""
        if not self._async_context:
            # Fall back to sync execution
            return self.model.train_step(x, y)

        batch_size = x.shape[0]

        # For small batches, use sync execution
        if batch_size < self.config.batch_threshold:
            return self.model.train_step(x, y)

        # Async execution
        return self._async_train_step(x, y)

    def _async_train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Async training step."""
        device = x.device
        batch_size = x.shape[0]

        input_proj = self.model.W_in(x)

        # Initialize activities
        for tile in self.model.graph.all_tiles:
            if tile.is_input:
                idx = self.model.graph.input_tile_ids.index(tile.id)
                start = idx * self.model.config.neurons_per_tile
                tile.activity = input_proj[:, start:start + tile.neurons].clone()
            else:
                tile.activity = torch.zeros(batch_size, tile.neurons, device=device)
            tile.prediction = None
            tile.error = None

        # Async relaxation loop
        for _ in range(self.model.config.inference_steps):
            self._async_relax_step(input_proj)

        # Task-driven learning (sync for now)
        return self._sync_learning(x, y)

    def _async_relax_step(self, input_proj: torch.Tensor):
        """One async relaxation step."""
        batch_size = input_proj.shape[0]
        device = input_proj.device

        # Compute priorities for all tiles
        priorities = self._compute_tile_priorities()

        # Submit prediction tasks
        for tile in self.model.graph.all_tiles:
            if not tile.is_input:
                task = TileTask(
                    tile_id=tile.id,
                    phase='predict',
                    input_data={'batch_size': batch_size, 'device': device},
                    priority=priorities.get(tile.id, 0.0)
                )
                self._scheduler.submit(task)

        # Process predictions
        processor = self._processors[0]
        results = self._scheduler.process_batch(self.model, processor)

        # Compute errors (sync, fast)
        self.model._compute_errors()

        # Submit update tasks
        for tile in self.model.graph.all_tiles:
            if not tile.is_input and tile.error is not None:
                task = TileTask(
                    tile_id=tile.id,
                    phase='update',
                    input_data={},
                    priority=priorities.get(tile.id, 0.0)
                )
                self._scheduler.submit(task)

        # Process updates
        results = self._scheduler.process_batch(self.model, processor)

    def _sync_learning(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Synchronous learning step (local Hebbian updates)."""
        # Use the model's built-in learning
        return self.model._train_step_pc(x, y)

    def _compute_tile_priorities(self) -> Dict[int, float]:
        """Compute priority scores for all tiles."""
        priorities = {}

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


class _AsyncContextManager:
    """Context manager for async execution."""

    def __init__(self, async_model: AsyncEquiTile):
        self.async_model = async_model

    def __enter__(self):
        self.async_model._async_context = True
        self.async_model._scheduler = TileScheduler(
            n_workers=self.async_model.config.n_workers,
            use_processes=self.async_model.config.use_processes
        )
        self.async_model._scheduler.start()
        self.async_model._processors = [
            TileProcessor(device) for device in self.async_model.devices
        ]
        return self.async_model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.async_model._scheduler.stop()
        self.async_model._scheduler = None
        self.async_model._processors = []
        self.async_model._async_context = False
        return False
