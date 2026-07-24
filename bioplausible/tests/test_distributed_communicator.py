import unittest
from dataclasses import dataclass
from typing import Dict
from typing import List

from bioplausible.equitile.distributed import DeviceAssignment
from bioplausible.equitile.distributed import TileCommunicator


# Mock TileGraph and TileState
@dataclass
class MockTile:
    id: int


class MockTileGraph:
    def __init__(self, edges):
        self.edges = edges
        self.tiles = {i: MockTile(i) for src, dst in edges for i in (src, dst)}

    def get_boundary_tiles(self, device_map: Dict[int, int]) -> Dict[int, List[int]]:
        # Mock implementation matching core.py
        boundary_map: Dict[int, List[int]] = {}
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


class TestTileCommunicator(unittest.TestCase):
    def test_find_boundary_tiles(self):
        # Setup graph: 0 -> 1 -> 2
        edges = [(0, 1), (1, 2)]
        graph = MockTileGraph(edges)

        # Setup assignments
        # Device 0: Tile 0
        # Device 1: Tile 1, 2
        assignments = [
            DeviceAssignment(device_id=0, device="cpu", tile_ids=[0], edge_ids=[]),
            DeviceAssignment(device_id=1, device="cpu", tile_ids=[1, 2], edge_ids=[]),
        ]

        comm = TileCommunicator(assignments, graph, backend="gloo")
        boundary = comm._boundary_tiles

        # Check Device 0 boundary
        # Tile 0 is connected to Tile 1 (on device 1)
        self.assertIn(0, boundary)
        self.assertEqual(len(boundary[0]), 1)
        self.assertEqual(boundary[0][0], (0, 1))  # (local, remote)

        # Check Device 1 boundary
        # Tile 1 is connected to Tile 0 (on device 0)
        # Tile 2 is connected to Tile 1 (same device) -> not boundary
        self.assertIn(1, boundary)
        self.assertEqual(len(boundary[1]), 1)
        self.assertEqual(boundary[1][0], (1, 0))  # (local, remote)


if __name__ == "__main__":
    unittest.main()
