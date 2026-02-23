from bioplausible.models.equitile.config import EquiTileConfig
from bioplausible.models.equitile.core import EquiTile
import torch

try:
    print("Testing EquiTileConfig with unknown kwargs...")
    config = EquiTileConfig(neurons_per_tile=64, unknown_arg=123)
    print("EquiTileConfig accepted unknown_arg")
except TypeError as e:
    print(f"EquiTileConfig raised TypeError: {e}")

try:
    print("\nTesting EquiTile init with unknown kwargs...")
    model = EquiTile(neurons_per_tile=64, unknown_arg=123)
    print("EquiTile init accepted unknown_arg")
except TypeError as e:
    print(f"EquiTile init raised TypeError: {e}")
except Exception as e:
    print(f"EquiTile init raised {type(e).__name__}: {e}")
