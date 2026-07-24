"""
EquiTile Topology: Graph Structure and Tile State
=================================================

Defines the graph topology and state containers for EquiTile models.
Moved from core.py to avoid circular dependencies and improve modularity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class TileState:
    """State for a single tile."""

    id: int
    neurons: int
    layer_id: int

    # Dynamic state
    activity: Tensor | None = None
    prediction: Tensor | None = None
    error: Tensor | None = None

    # Metadata
    is_input: bool = False
    is_output: bool = False
    pos_x: float = 0.0
    pos_y: float = 0.0

    # Connectivity
    fwd_neighbors: list[int] = field(default_factory=list)
    bwd_neighbors: list[int] = field(default_factory=list)


class TileGraph:
    """Manages tile connectivity and state."""

    def __init__(self) -> None:
        self.tiles: dict[int, TileState] = {}
        self.edges: list[tuple[int, int]] = []
        self._edge_set: set[tuple[int, int]] = set()
        self.layer_ids: list[list[int]] = []
        self.input_tile_ids: list[int] = []
        self.output_tile_ids: list[int] = []

    def build_layered(
        self,
        input_dim: int,
        output_dim: int,
        neurons_per_tile: int,
        num_hidden_layers: int,
        tiles_per_layer: int = 1,
        use_skip_connections: bool = False,
    ) -> None:
        hidden_dim = neurons_per_tile * tiles_per_layer
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        total_layers = len(dims)

        tile_id = 0

        for layer_idx, dim in enumerate(dims):
            n_tiles = math.ceil(dim / neurons_per_tile)
            layer_tile_ids: list[int] = []

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

        # Add skip connections (every 2 layers) if enabled
        if use_skip_connections and len(self.layer_ids) > 2:
            for layer_idx in range(len(self.layer_ids) - 2):
                for src_id in self.layer_ids[layer_idx]:
                    for dst_id in self.layer_ids[layer_idx + 2]:
                        # Only add if not already connected
                        self._add_edge(src_id, dst_id)

    def build_custom(
        self,
        n_tiles: int,
        neurons_per_tile: int,
        edges: list[tuple[int, int]],
        input_ids: list[int],
        output_ids: list[int],
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
    def all_tiles(self) -> list[TileState]:
        return [self.tiles[i] for i in sorted(self.tiles.keys())]

    def get_boundary_tiles(self, device_map: dict[int, int]) -> dict[int, list[int]]:
        """Identify boundary tiles that connect to different devices.

        Parameters
        ----------
        device_map : dict
            Mapping from tile_id to device_id

        Returns
        -------
        dict
            Mapping from tile_id to list of neighbor tile_ids on different devices
        """
        boundary_map: dict[int, list[int]] = {}
        for src, dst in self.edges:
            src_dev = device_map.get(src)
            dst_dev = device_map.get(dst)

            if src_dev is not None and dst_dev is not None and src_dev != dst_dev:
                # src is boundary
                if src not in boundary_map:
                    boundary_map[src] = []
                boundary_map[src].append(dst)

                # dst is boundary
                if dst not in boundary_map:
                    boundary_map[dst] = []
                boundary_map[dst].append(src)

        return boundary_map
