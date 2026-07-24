import unittest

import torch

from bioplausible.equitile.core import EquiTile
from bioplausible.equitile.core import EquiTileConfig
from bioplausible.equitile.distributed import DistributedConfig
from bioplausible.equitile.distributed import DistributedEquiTile


class TestDistributedEquiTile(unittest.TestCase):
    def setUp(self):
        self.config = EquiTileConfig(
            neurons_per_tile=16, num_layers=3, tiles_per_layer=2, mode="pc"
        )
        self.model = EquiTile(config=self.config, input_dim=8, output_dim=4)

        # CPU config for testing
        self.dist_config = DistributedConfig(
            device_ids=[],  # Triggers CPU fallback if no cuda
            mixed_precision=False,  # Disable mixed precision for CPU test
        )
        self.dist_model = DistributedEquiTile(self.model, self.dist_config)

    def test_init_and_forward(self):
        """Test initialization and distributed forward (emulated)."""
        x = torch.randn(2, 8)
        y = torch.randint(0, 4, (2,))

        stats = self.dist_model.train_step(x, y)
        self.assertIn("loss", stats)
        self.assertIn("accuracy", stats)

    def test_grow_tile(self):
        """Test growing a tile in distributed setting."""
        self.dist_model.growth_config.enabled = True

        # Pick a parent tile from assignments
        parent_id = self.dist_model.assignments[0].tile_ids[0]

        new_id = self.dist_model.grow_tile(parent_id)
        self.assertNotEqual(new_id, -1)
        self.assertIn(new_id, self.model.graph.tiles)

        # Check assignment update
        found = False
        for assignment in self.dist_model.assignments:
            if new_id in assignment.tile_ids:
                found = True
                break
        self.assertTrue(found, "New tile not assigned to any device")

    def test_prune_tile(self):
        """Test pruning a tile."""
        self.dist_model.growth_config.enabled = True
        new_id = self.dist_model.grow_tile(self.dist_model.assignments[0].tile_ids[0])

        self.assertTrue(self.dist_model.prune_tile(new_id))
        self.assertNotIn(new_id, self.model.graph.tiles)

        # Check assignment update
        found = False
        for assignment in self.dist_model.assignments:
            if new_id in assignment.tile_ids:
                found = True
        self.assertFalse(found, "Pruned tile still in assignments")


if __name__ == "__main__":
    unittest.main()
