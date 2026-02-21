"""
EquiTile Dynamics: Tile Growth and Pruning
==========================================

Dynamic tile architecture that adapts during training:
- TileGrowthManager: Manages tile lifecycle
- TileGrowthConfig: Configuration
- DynamicEquiTile: Wrapper with dynamic architecture
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .core import EquiTile, TileState, EdgeParams


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TileGrowthConfig:
    """Tile growth and pruning configuration.

    Attributes
    ----------
    growth_enabled : bool
        Enable tile growth
    prune_enabled : bool
        Enable tile pruning
    growth_threshold : float
        Error threshold for adding tiles
    prune_threshold : float
        Error threshold for removing tiles
    growth_cooldown : int
        Steps between growth operations
    prune_cooldown : int
        Steps between pruning operations
    max_tiles : int
        Maximum number of tiles
    min_tiles : int
        Minimum number of tiles
    error_ema_decay : float
        EMA decay for error tracking
    min_age_for_modify : int
        Minimum tile age before modification
    """
    growth_enabled: bool = True
    prune_enabled: bool = True
    growth_threshold: float = 0.5
    prune_threshold: float = 0.05
    growth_cooldown: int = 100
    prune_cooldown: int = 200
    max_tiles: int = 100
    min_tiles: int = 2
    error_ema_decay: float = 0.95
    min_age_for_modify: int = 50


@dataclass
class TileMetrics:
    """Metrics for a single tile."""
    tile_id: int
    error_mean: float = 0.0
    error_max: float = 0.0
    importance: float = 0.0
    age: int = 0


# =============================================================================
# Tile Growth Manager
# =============================================================================

class TileGrowthManager:
    """Manages tile growth and pruning lifecycle.

    Tracks tile metrics and decides when to add/remove tiles.

    Parameters
    ----------
    config : TileGrowthConfig, optional
        Growth configuration
    """

    def __init__(self, config: Optional[TileGrowthConfig] = None):
        self.config = config or TileGrowthConfig()
        self.metrics: Dict[int, TileMetrics] = {}
        self.error_ema: Dict[int, float] = {}
        self._step_count = 0
        self._last_growth_step = -self.config.growth_cooldown
        self._last_prune_step = -self.config.prune_cooldown

    def update_metrics(self, model: 'EquiTile'):
        """Update metrics for all tiles.

        Parameters
        ----------
        model : EquiTile
            The model
        """
        for i, tile in enumerate(model.graph.all_tiles):
            if tile.id not in self.metrics:
                self.metrics[tile.id] = TileMetrics(tile_id=tile.id)

            metrics = self.metrics[tile.id]

            # Update error statistics
            if tile.error is not None:
                error_norm = tile.error.norm(p=2, dim=-1).mean().item()
                metrics.error_mean = (
                    self.config.error_ema_decay * metrics.error_mean +
                    (1 - self.config.error_ema_decay) * error_norm
                )
                metrics.error_max = max(metrics.error_max, error_norm)

            # Update importance
            importance = torch.sigmoid(model.tile_importance[i]).item()
            metrics.importance = importance

            # Update age
            metrics.age += 1

            # Track EMA
            self.error_ema[tile.id] = (
                self.config.error_ema_decay * self.error_ema.get(tile.id, 0.0) +
                (1 - self.config.error_ema_decay) * metrics.error_mean
            )

    def should_grow(self, model: 'EquiTile') -> Optional[int]:
        """Check if we should add a tile.

        Parameters
        ----------
        model : EquiTile
            The model

        Returns
        -------
        int, optional
            Parent tile ID or None
        """
        if not self.config.growth_enabled:
            return None

        if self._step_count - self._last_growth_step < self.config.growth_cooldown:
            return None

        n_tiles = len(model.graph.tiles)
        if n_tiles >= self.config.max_tiles:
            return None

        # Find tile with highest persistent error
        candidates = []
        for tile_id, error in self.error_ema.items():
            tile = model.graph.tiles.get(tile_id)
            if tile is None or tile.is_input or tile.is_output:
                continue

            metrics = self.metrics.get(tile_id)
            if metrics is None or metrics.age < self.config.min_age_for_modify:
                continue

            if error > self.config.growth_threshold:
                candidates.append((tile_id, error))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def should_prune(self, model: 'EquiTile') -> Optional[int]:
        """Check if we should remove a tile.

        Parameters
        ----------
        model : EquiTile
            The model

        Returns
        -------
        int, optional
            Tile ID or None
        """
        if not self.config.prune_enabled:
            return None

        if self._step_count - self._last_prune_step < self.config.prune_cooldown:
            return None

        n_tiles = len(model.graph.tiles)
        if n_tiles <= self.config.min_tiles:
            return None

        # Find tile with lowest persistent error
        candidates = []
        for tile_id, error in self.error_ema.items():
            tile = model.graph.tiles.get(tile_id)
            if tile is None or tile.is_input or tile.is_output:
                continue

            metrics = self.metrics.get(tile_id)
            if metrics is None or metrics.age < self.config.min_age_for_modify:
                continue

            if error < self.config.prune_threshold:
                candidates.append((tile_id, error))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def grow_tile(self, model: 'EquiTile', parent_id: int) -> int:
        """Add a new tile as a child of an existing tile.

        Parameters
        ----------
        model : EquiTile
            The model
        parent_id : int
            Parent tile ID

        Returns
        -------
        int
            New tile ID
        """
        parent = model.graph.tiles[parent_id]
        new_id = max(model.graph.tiles.keys()) + 1

        # Create new tile
        new_tile = TileState(
            id=new_id,
            neurons=parent.neurons,
            layer_id=parent.layer_id,
            is_input=False,
            is_output=False,
        )

        model.graph.tiles[new_id] = new_tile

        # Connect to parent's forward neighbors
        for dst_id in parent.fwd_neighbors:
            edge = model.graph.edges.get((parent_id, dst_id))
            if edge is not None:
                model.graph.edges[(new_id, dst_id)] = EdgeParams(
                    src_id=new_id,
                    dst_id=dst_id,
                    weight=edge.weight.clone() * 0.5,
                    bias=edge.bias.clone() * 0.5 if edge.bias is not None else None,
                )
                new_tile.fwd_neighbors.append(dst_id)

                if new_id not in model.graph.tiles[dst_id].bwd_neighbors:
                    model.graph.tiles[dst_id].bwd_neighbors.append(new_id)

        # Add lateral edge from parent to new tile
        model.graph.edges[(parent_id, new_id)] = EdgeParams(
            src_id=parent_id,
            dst_id=new_id,
            weight=torch.randn(parent.neurons, new_tile.neurons) * 0.01,
            bias=torch.zeros(new_tile.neurons),
        )
        parent.fwd_neighbors.append(new_id)
        new_tile.bwd_neighbors.append(parent_id)

        # Extend importance parameters
        with torch.no_grad():
            old_importance = model.tile_importance.clone()
            model.tile_importance = torch.nn.Parameter(
                torch.cat([old_importance, torch.ones(1)])
            )

        self._last_growth_step = self._step_count
        return new_id

    def prune_tile(self, model: 'EquiTile', tile_id: int) -> bool:
        """Remove a tile and its connections.

        Parameters
        ----------
        model : EquiTile
            The model
        tile_id : int
            Tile to remove

        Returns
        -------
        bool
            Whether tile was pruned
        """
        tile = model.graph.tiles.get(tile_id)
        if tile is None or tile.is_input or tile.is_output:
            return False

        # Remove all edges connected to this tile
        edges_to_remove = [
            key for key in list(model.graph.edges.keys())
            if tile_id in key
        ]

        for edge_key in edges_to_remove:
            src_id, dst_id = edge_key

            if src_id != tile_id and tile_id in model.graph.tiles[src_id].fwd_neighbors:
                model.graph.tiles[src_id].fwd_neighbors.remove(tile_id)
            if dst_id != tile_id and tile_id in model.graph.tiles[dst_id].bwd_neighbors:
                model.graph.tiles[dst_id].bwd_neighbors.remove(tile_id)

            del model.graph.edges[edge_key]

        # Remove tile
        del model.graph.tiles[tile_id]

        # Update importance parameters
        with torch.no_grad():
            if tile_id in model.graph.tiles:
                tile_idx = list(model.graph.tiles.keys()).index(tile_id)
                mask = torch.ones(len(model.tile_importance), dtype=torch.bool)
                mask[tile_idx] = False
                model.tile_importance = torch.nn.Parameter(model.tile_importance[mask])

        # Clean up metrics
        if tile_id in self.metrics:
            del self.metrics[tile_id]
        if tile_id in self.error_ema:
            del self.error_ema[tile_id]

        self._last_prune_step = self._step_count
        return True

    def step(self, model: 'EquiTile') -> Dict[str, int]:
        """Perform one step of tile dynamics.

        Parameters
        ----------
        model : EquiTile
            The model

        Returns
        -------
        Dict[str, int]
            Modification counts
        """
        self._step_count += 1
        self.update_metrics(model)

        stats = {"grown": 0, "pruned": 0}

        # Check for growth
        parent_id = self.should_grow(model)
        if parent_id is not None:
            self.grow_tile(model, parent_id)
            stats["grown"] = 1

        # Check for pruning
        prune_id = self.should_prune(model)
        if prune_id is not None:
            if self.prune_tile(model, prune_id):
                stats["pruned"] = 1

        return stats

    def reset(self):
        """Reset all state."""
        self.metrics.clear()
        self.error_ema.clear()
        self._step_count = 0
        self._last_growth_step = -self.config.growth_cooldown
        self._last_prune_step = -self.config.prune_cooldown


# =============================================================================
# Dynamic EquiTile
# =============================================================================

@dataclass
class DynamicEquiTileConfig:
    """Dynamic tile architecture configuration."""
    growth: TileGrowthConfig = field(default_factory=TileGrowthConfig)
    track_history: bool = True
    max_history: int = 1000


class DynamicEquiTile:
    """EquiTile with dynamic tile architecture.

    Automatically grows and prunes tiles during training based on error signals.

    Parameters
    ----------
    model : EquiTile
        Base EquiTile model
    config : DynamicEquiTileConfig, optional
        Dynamic configuration

    Examples
    --------
    >>> model = EquiTile(...)
    >>> dynamic = DynamicEquiTile(model)
    >>> for X, y in dataloader:
    ...     stats = model.train_step(X, y)
    ...     mods = dynamic.step()
    ...     if mods['grown'] > 0:
    ...         print(f"Grew {mods['grown']} tiles")
    """

    def __init__(
        self,
        model: 'EquiTile',
        config: Optional[DynamicEquiTileConfig] = None,
    ):
        self.model = model
        self.config = config or DynamicEquiTileConfig()

        self.growth_manager = TileGrowthManager(self.config.growth)
        self.tile_modified = False
        self._history: List[Dict] = [] if self.config.track_history else None

    def step(self) -> Dict[str, int]:
        """Perform one step of tile dynamics.

        Returns
        -------
        Dict[str, int]
            Modification counts
        """
        stats = self.growth_manager.step(self.model)
        self.tile_modified = stats["grown"] > 0 or stats["pruned"] > 0

        # Track history
        if self._history is not None:
            self._history.append({
                "step": self.growth_manager._step_count,
                "n_tiles": len(self.model.graph.tiles),
                "n_edges": len(self.model.graph.edges),
                **stats,
            })

            if len(self._history) > self.config.max_history:
                self._history.pop(0)

        return stats

    def get_tile_metrics(self) -> Dict[int, TileMetrics]:
        """Get metrics for all tiles.

        Returns
        -------
        Dict[int, TileMetrics]
            Tile metrics
        """
        return self.growth_manager.metrics

    def get_error_distribution(self) -> Dict[str, float]:
        """Get error distribution statistics.

        Returns
        -------
        Dict[str, float]
            Error statistics
        """
        errors = list(self.growth_manager.error_ema.values())

        if not errors:
            return {}

        mean_error = sum(errors) / len(errors)
        return {
            "mean": mean_error,
            "std": (sum((e - mean_error)**2 for e in errors) / len(errors)) ** 0.5,
            "min": min(errors),
            "max": max(errors),
            "hot_tiles": sum(1 for e in errors if e > self.config.growth.growth_threshold),
            "cold_tiles": sum(1 for e in errors if e < self.config.growth.prune_threshold),
        }

    def get_history(self) -> List[Dict]:
        """Get modification history.

        Returns
        -------
        List[Dict]
            History of modifications
        """
        return self._history or []

    def reset(self):
        """Reset dynamic state."""
        self.growth_manager.reset()
        self.tile_modified = False
        if self._history is not None:
            self._history.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_dynamic_model(
    neurons_per_tile: int = 32,
    num_layers: int = 3,
    tiles_per_layer: int = 2,
    input_dim: int = 64,
    output_dim: int = 4,
    growth_enabled: bool = True,
    prune_enabled: bool = True,
    **kwargs,
):
    """Create EquiTile with dynamic tile architecture.

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
    growth_enabled : bool
        Enable tile growth
    prune_enabled : bool
        Enable tile pruning

    Returns
    -------
    Tuple[EquiTile, DynamicEquiTile]
        Model and dynamic wrapper
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

    growth_config = TileGrowthConfig(
        growth_enabled=growth_enabled,
        prune_enabled=prune_enabled,
        **kwargs
    )

    dynamic = DynamicEquiTile(
        model,
        config=DynamicEquiTileConfig(growth=growth_config),
    )

    return model, dynamic
