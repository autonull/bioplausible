"""
EquiTile Dynamics: Production Tile Growth/Pruning
=================================================

Dynamic tile architecture that adapts during training:
- Add tiles when error is persistently high
- Remove tiles when error is persistently low
- Merge similar tiles
- Split overloaded tiles

Key Components
--------------
- TileGrowthManager: Manages tile lifecycle
- TilePruner: Prunes underutilized tiles
- TileMerger: Merges similar tiles
- TileSplitter: Splits overloaded tiles
- DynamicEquiTile: Full dynamic architecture
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .config import TileGrowthConfig, DynamicEquiTileConfig
from .core import TileState, EdgeParams

if TYPE_CHECKING:
    from .core import EquiTile


@dataclass
class TileMetrics:
    """Metrics for a single tile."""
    tile_id: int
    error_mean: float = 0.0
    error_std: float = 0.0
    error_max: float = 0.0
    activity_mean: float = 0.0
    activity_std: float = 0.0
    importance: float = 0.0
    update_count: int = 0
    age: int = 0  # Steps since creation
    last_modified: int = 0


class TileGrowthManager:
    """Manages tile growth and pruning lifecycle.

    Tracks tile metrics and decides when to add/remove tiles.
    """

    def __init__(self, config: TileGrowthConfig = None):
        self.config = config or TileGrowthConfig()
        self.metrics: Dict[int, TileMetrics] = {}
        self.error_ema: Dict[int, float] = {}
        self._step_count = 0
        self._last_growth_step = -self.config.growth_cooldown
        self._last_prune_step = -self.config.prune_cooldown
        self._last_merge_step = -self.config.merge_cooldown
        self._last_split_step = -self.config.split_cooldown

    def update_metrics(self, model: 'EquiTile'):
        """Update metrics for all tiles."""
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
        """Check if we should add a tile. Returns parent tile ID or None."""
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
                # Check layer constraint
                layer_tiles = sum(
                    1 for t in model.graph.all_tiles if t.layer_id == tile.layer_id
                )
                if layer_tiles < self.config.max_tiles_per_layer:
                    candidates.append((tile_id, error))

        if candidates:
            # Return tile with highest error
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def should_prune(self, model: 'EquiTile') -> Optional[int]:
        """Check if we should remove a tile. Returns tile ID or None."""
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
                # Check layer constraint
                layer_tiles = sum(
                    1 for t in model.graph.all_tiles if t.layer_id == tile.layer_id
                )
                if layer_tiles > self.config.min_tiles_per_layer:
                    candidates.append((tile_id, error))

        if candidates:
            # Return tile with lowest error
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def step(self, model: 'EquiTile') -> Dict[str, int]:
        """Perform one step of tile dynamics.

        Returns dict with counts of tiles grown/pruned/merged/split.
        """
        self._step_count += 1
        self.update_metrics(model)

        stats = {"grown": 0, "pruned": 0, "merged": 0, "split": 0}

        # Check for growth
        parent_id = self.should_grow(model)
        if parent_id is not None:
            self.grow_tile(model, parent_id)
            stats["grown"] = 1
            self._last_growth_step = self._step_count

        # Check for pruning
        prune_id = self.should_prune(model)
        if prune_id is not None:
            if self.prune_tile(model, prune_id):
                stats["pruned"] = 1
                self._last_prune_step = self._step_count

        return stats

    def grow_tile(self, model: 'EquiTile', parent_id: int) -> int:
        """Add a new tile as a child of an existing tile."""
        parent = model.graph.tiles[parent_id]
        new_id = max(model.graph.tiles.keys()) + 1

        # Create new tile with same neurons as parent
        new_tile = TileState(
            id=new_id,
            neurons=parent.neurons,
            layer_id=parent.layer_id,
            is_input=False,
            is_output=False,
            pos_x=parent.pos_x + 0.05,  # Slightly offset
            pos_y=parent.pos_y,
        )

        model.graph.tiles[new_id] = new_tile

        # Connect to parent's neighbors
        for dst_id in parent.fwd_neighbors:
            # Add edge from new tile to parent's forward neighbors
            edge = model.graph.edges.get((parent_id, dst_id))
            if edge is not None:
                model.graph.edges[(new_id, dst_id)] = EdgeParams(
                    src_id=new_id,
                    dst_id=dst_id,
                    weight=edge.weight.clone() * 0.5,  # Initialize with half weight
                    bias=edge.bias.clone() * 0.5 if edge.bias is not None else None,
                )
                new_tile.fwd_neighbors.append(dst_id)

                # Update dst's backward neighbors
                if new_id not in model.graph.tiles[dst_id].bwd_neighbors:
                    model.graph.tiles[dst_id].bwd_neighbors.append(new_id)

        # Add edge from parent to new tile (lateral connection)
        model.graph.edges[(parent_id, new_id)] = EdgeParams(
            src_id=parent_id,
            dst_id=new_id,
            weight=torch.randn(parent.neurons, new_tile.neurons) * 0.01,
            bias=torch.zeros(new_tile.neurons),
        )
        parent.fwd_neighbors.append(new_id)
        new_tile.bwd_neighbors.append(parent_id)

        # Initialize importance
        with torch.no_grad():
            # Extend tile importance parameters
            old_tile_importance = model.tile_importance.clone()
            model.tile_importance = torch.nn.Parameter(
                torch.cat([old_tile_importance, torch.ones(1)])
            )

            # Extend edge importance parameters
            # We added some number of edges. We need to match the new length of graph.edges
            current_edge_count = len(model.graph.edges)
            old_edge_importance = model.edge_importance.clone()
            added_edges = current_edge_count - len(old_edge_importance)

            if added_edges > 0:
                model.edge_importance = torch.nn.Parameter(
                    torch.cat([old_edge_importance, torch.ones(added_edges)])
                )

        # Reset optimizers to include new parameters
        model.reset_optimizers()

        print(f"  Grew tile {new_id} from parent {parent_id}")
        return new_id

    def prune_tile(self, model: 'EquiTile', tile_id: int) -> bool:
        """Remove a tile and its connections."""
        tile = model.graph.tiles.get(tile_id)
        if tile is None or tile.is_input or tile.is_output:
            return False

        # Identify edges to remove and their indices
        edges_to_remove_keys = [
            key for key in list(model.graph.edges.keys())
            if tile_id in key
        ]

        # Determine edge indices to remove
        edge_keys = list(model.graph.edges.keys())
        edge_indices_to_remove = {edge_keys.index(k) for k in edges_to_remove_keys}

        # Determine tile index to remove
        sorted_tile_ids = sorted(model.graph.tiles.keys())
        tile_idx_to_remove = sorted_tile_ids.index(tile_id) if tile_id in sorted_tile_ids else -1

        # Remove edges from graph
        for edge_key in edges_to_remove_keys:
            src_id, dst_id = edge_key

            # Update neighbor lists
            if src_id != tile_id and tile_id in model.graph.tiles[src_id].fwd_neighbors:
                model.graph.tiles[src_id].fwd_neighbors.remove(tile_id)
            if dst_id != tile_id and tile_id in model.graph.tiles[dst_id].bwd_neighbors:
                model.graph.tiles[dst_id].bwd_neighbors.remove(tile_id)

            del model.graph.edges[edge_key]

        # Remove tile from graph
        del model.graph.tiles[tile_id]

        # Update importance parameters
        with torch.no_grad():
            # Update edge importance
            if len(edge_indices_to_remove) > 0:
                edge_mask = torch.ones(len(model.edge_importance), dtype=torch.bool)
                for idx in edge_indices_to_remove:
                    edge_mask[idx] = False
                model.edge_importance = torch.nn.Parameter(
                    model.edge_importance[edge_mask]
                )

            # Update tile importance
            if tile_idx_to_remove >= 0:
                tile_mask = torch.ones(len(model.tile_importance), dtype=torch.bool)
                tile_mask[tile_idx_to_remove] = False
                model.tile_importance = torch.nn.Parameter(
                    model.tile_importance[tile_mask]
                )

        # Clean up metrics
        if tile_id in self.metrics:
            del self.metrics[tile_id]
        if tile_id in self.error_ema:
            del self.error_ema[tile_id]

        # Reset optimizers to remove pruned parameters
        model.reset_optimizers()

        print(f"  Pruned tile {tile_id}")
        return True

    def reset(self):
        """Reset all state."""
        self.metrics.clear()
        self.error_ema.clear()
        self._step_count = 0
        self._last_growth_step = -self.config.growth_cooldown
        self._last_prune_step = -self.config.prune_cooldown


class TileMerger:
    """Merges similar tiles to reduce redundancy."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def find_similar_tiles(
        self,
        model: 'EquiTile',
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of similar tiles.

        Returns list of (tile1_id, tile2_id, similarity) tuples.
        """
        similar_pairs = []
        tiles = model.graph.all_tiles

        for i, tile1 in enumerate(tiles):
            if tile1.is_input or tile1.is_output:
                continue

            for tile2 in tiles[i+1:]:
                if tile2.is_input or tile2.is_output:
                    continue

                # Must be in same layer
                if tile1.layer_id != tile2.layer_id:
                    continue

                # Compute similarity from activities
                if tile1.activity is not None and tile2.activity is not None:
                    # Flatten activities
                    a1 = tile1.activity.mean(dim=0)
                    a2 = tile2.activity.mean(dim=0)

                    # Cosine similarity
                    sim = F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0)).item()

                    if sim > self.threshold:
                        similar_pairs.append((tile1.id, tile2.id, sim))

        return similar_pairs

    def merge_tiles(
        self,
        model: 'EquiTile',
        tile1_id: int,
        tile2_id: int,
    ) -> int:
        """Merge two tiles into one.

        Returns ID of merged tile.
        """
        tile1 = model.graph.tiles[tile1_id]
        tile2 = model.graph.tiles[tile2_id]

        # Create merged tile
        merged_id = max(model.graph.tiles.keys()) + 1

        merged_tile = TileState(
            id=merged_id,
            neurons=tile1.neurons,  # Same as originals
            layer_id=tile1.layer_id,
            is_input=False,
            is_output=False,
        )

        # Average activities
        if tile1.activity is not None and tile2.activity is not None:
            merged_tile.activity = (tile1.activity + tile2.activity) / 2

        model.graph.tiles[merged_id] = merged_tile

        # Combine edges
        for dst_id in set(tile1.fwd_neighbors) | set(tile2.fwd_neighbors):
            edge1 = model.graph.edges.get((tile1_id, dst_id))
            edge2 = model.graph.edges.get((tile2_id, dst_id))

            if edge1 is not None and edge2 is not None:
                # Average weights
                merged_weight = (edge1.weight + edge2.weight) / 2
                merged_bias = None
                if edge1.bias is not None and edge2.bias is not None:
                    merged_bias = (edge1.bias + edge2.bias) / 2

                model.graph.edges[(merged_id, dst_id)] = EdgeParams(
                    src_id=merged_id,
                    dst_id=dst_id,
                    weight=merged_weight,
                    bias=merged_bias,
                )
            elif edge1 is not None:
                model.graph.edges[(merged_id, dst_id)] = EdgeParams(
                    src_id=merged_id,
                    dst_id=dst_id,
                    weight=edge1.weight.clone(),
                    bias=edge1.bias.clone() if edge1.bias is not None else None,
                )
            elif edge2 is not None:
                model.graph.edges[(merged_id, dst_id)] = EdgeParams(
                    src_id=merged_id,
                    dst_id=dst_id,
                    weight=edge2.weight.clone(),
                    bias=edge2.bias.clone() if edge2.bias is not None else None,
                )

            merged_tile.fwd_neighbors.append(dst_id)

        # Remove old tiles
        # (Would use TileGrowthManager.prune_tile)

        return merged_id


class TileSplitter:
    """Splits overloaded tiles into multiple tiles."""

    def split_tile(
        self,
        model: 'EquiTile',
        tile_id: int,
        n_splits: int = 2,
    ) -> List[int]:
        """Split a tile into multiple tiles.

        Returns list of new tile IDs.
        """
        tile = model.graph.tiles[tile_id]
        new_ids = []

        neurons_per_split = tile.neurons // n_splits

        for i in range(n_splits):
            new_id = max(model.graph.tiles.keys()) + 1
            new_ids.append(new_id)

            start_neuron = i * neurons_per_split
            end_neuron = start_neuron + neurons_per_split

            new_tile = TileState(
                id=new_id,
                neurons=neurons_per_split,
                layer_id=tile.layer_id,
                is_input=False,
                is_output=False,
            )

            # Initialize with subset of activity
            if tile.activity is not None:
                new_tile.activity = tile.activity[:, start_neuron:end_neuron].clone()

            model.graph.tiles[new_id] = new_tile

            # Copy edges with subset of weights
            for dst_id in tile.fwd_neighbors:
                edge = model.graph.edges.get((tile_id, dst_id))
                if edge is not None:
                    model.graph.edges[(new_id, dst_id)] = EdgeParams(
                        src_id=new_id,
                        dst_id=dst_id,
                        weight=edge.weight[start_neuron:end_neuron, :].clone(),
                        bias=edge.bias.clone() if edge.bias is not None else None,
                    )
                    new_tile.fwd_neighbors.append(dst_id)

        # Remove original tile
        # (Would use TileGrowthManager.prune_tile)

        return new_ids


class DynamicEquiTile:
    """EquiTile with dynamic tile architecture.

    Automatically grows, prunes, merges, and splits tiles during training
    based on error signals and utilization.

    Usage:
        model = EquiTile(...)
        dynamic = DynamicEquiTile(model)

        for X, y in dataloader:
            stats = model.train_step(X, y)
            dynamic.step()  # Check for growth/pruning

            if dynamic.tile_modified:
                print(f"Tiles: {len(model.graph.tiles)}")
    """

    def __init__(
        self,
        model: 'EquiTile',
        config: DynamicEquiTileConfig = None,
    ):
        self.model = model
        self.config = config or DynamicEquiTileConfig()

        self.growth_manager = TileGrowthManager(self.config.growth)
        self.merger = TileMerger(threshold=self.config.growth.merge_threshold)
        self.splitter = TileSplitter()

        self.tile_modified = False
        self._history: List[Dict] = [] if self.config.track_history else None

    def step(self) -> Dict[str, int]:
        """Perform one step of tile dynamics.

        Returns dict with modification counts.
        """
        stats = self.growth_manager.step(self.model)
        self.tile_modified = stats["grown"] > 0 or stats["pruned"] > 0

        # Check for merging
        if self.config.merge_enabled and stats == {"grown": 0, "pruned": 0}:
            similar = self.merger.find_similar_tiles(self.model)
            if similar:
                tile1, tile2, _ = similar[0]
                self.merger.merge_tiles(self.model, tile1, tile2)
                stats["merged"] = 1
                self.tile_modified = True

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
        """Get metrics for all tiles."""
        return self.growth_manager.metrics

    def get_error_distribution(self) -> Dict[str, float]:
        """Get error distribution statistics."""
        errors = list(self.growth_manager.error_ema.values())

        if not errors:
            return {}

        return {
            "mean": sum(errors) / len(errors),
            "std": (sum((e - sum(errors)/len(errors))**2 for e in errors) / len(errors)) ** 0.5,
            "min": min(errors),
            "max": max(errors),
            "hot_tiles": sum(1 for e in errors if e > self.config.growth.growth_threshold),
            "cold_tiles": sum(1 for e in errors if e < self.config.growth.prune_threshold),
        }

    def get_history(self) -> List[Dict]:
        """Get modification history."""
        return self._history or []

    def reset(self):
        """Reset dynamic state."""
        self.growth_manager.reset()
        self.tile_modified = False
        if self._history is not None:
            self._history.clear()


def create_dynamic_model(
    neurons_per_tile: int,
    num_layers: int,
    tiles_per_layer: int,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> Tuple['EquiTile', DynamicEquiTile]:
    """Create EquiTile with dynamic tile architecture.

    Usage:
        model, dynamic = create_dynamic_model(
            neurons_per_tile=64,
            num_layers=4,
            tiles_per_layer=4,
            input_dim=784,
            output_dim=10,
            growth_enabled=True,
            prune_enabled=True,
        )
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
        growth_enabled=kwargs.get('growth_enabled', True),
        prune_enabled=kwargs.get('prune_enabled', True),
        merge_enabled=kwargs.get('merge_enabled', False),
        split_enabled=kwargs.get('split_enabled', False),
        max_tiles=kwargs.get('max_tiles', 100),
        growth_threshold=kwargs.get('growth_threshold', 0.5),
        prune_threshold=kwargs.get('prune_threshold', 0.05),
    )

    dynamic = DynamicEquiTile(
        model,
        config=DynamicEquiTileConfig(growth=growth_config),
    )

    return model, dynamic
