"""
EquiTile: Scalable Local-Learning Architecture
==============================================

A production-ready, tile-based local learning framework featuring:
- Tile-based parallel architecture
- Local Hebbian weight updates (no global backprop)
- Multi-GPU support with NCCL
- Mixed precision training
- Dynamic tile growth/pruning
- Enhanced EP with LayerNorm and curriculum learning

Quick Start
-----------
>>> from bioplausible.models.equitile import EquiTile
>>> model = EquiTile(
...     neurons_per_tile=64,
...     num_layers=4,
...     tiles_per_layer=4,
...     input_dim=784,
...     output_dim=10,
... )
>>> for X, y in dataloader:
...     stats = model.train_step(X, y)

Modules
-------
core : Core EquiTile implementation
config : Configuration classes
"""

from .config import (
    EquiTileConfig,
    create_production_config,
    create_research_config,
    create_fast_config,
)

from .core import EquiTile, TileGraph, TileState, EdgeParams

__all__ = [
    # Core
    "EquiTile",
    "TileGraph",
    "TileState",
    "EdgeParams",
    
    # Config
    "EquiTileConfig",
    "create_production_config",
    "create_research_config",
    "create_fast_config",
]

__version__ = "1.0.0"
