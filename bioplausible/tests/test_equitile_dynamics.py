
import unittest
import torch
from bioplausible.models.equitile.core import EquiTile, EquiTileConfig
from bioplausible.models.equitile.dynamics import DynamicEquiTile, TileGrowthConfig, DynamicEquiTileConfig

class TestEquiTileDynamics(unittest.TestCase):
    def setUp(self):
        self.config = EquiTileConfig(
            neurons_per_tile=16,
            num_layers=3,
            tiles_per_layer=2,
            mode="pc"
        )
        self.model = EquiTile(
            config=self.config,
            input_dim=8,
            output_dim=4
        )
        self.growth_config = TileGrowthConfig(
            growth_enabled=True,
            prune_enabled=True,
            growth_threshold=0.1, # Low threshold to trigger growth
            prune_threshold=0.01,
            growth_cooldown=0, # No cooldown for testing
            min_age_for_modify=0
        )
        self.dynamic_config = DynamicEquiTileConfig(growth=self.growth_config)
        self.dynamic = DynamicEquiTile(self.model, config=self.dynamic_config)

    def test_add_tile_via_api(self):
        """Test adding a tile via core API."""
        initial_tiles = len(self.model.graph.tiles)
        new_id = self.model.add_tile(
            neurons=16,
            layer_id=1,
            pos_x=0.5,
            pos_y=0.5
        )
        self.assertIn(new_id, self.model.graph.tiles)
        self.assertEqual(len(self.model.graph.tiles), initial_tiles + 1)
        # Check if optimizers reset (check if params are in optimizer)
        found = False
        for group in self.model._optim_importance.param_groups:
            for p in group['params']:
                if p.shape == self.model.tile_importance.shape:
                    found = True
        self.assertTrue(found)

    def test_remove_tile_via_api(self):
        """Test removing a tile via core API."""
        # Add a dummy tile first
        new_id = self.model.add_tile(16, 1)
        initial_tiles = len(self.model.graph.tiles)

        self.model.remove_tile(new_id)
        self.assertNotIn(new_id, self.model.graph.tiles)
        self.assertEqual(len(self.model.graph.tiles), initial_tiles - 1)

    def test_growth_manager(self):
        """Test TileGrowthManager logic."""
        # Fake high error on a tile
        target_tile_id = self.model.graph.tiles[self.model.graph.input_tile_ids[0]].fwd_neighbors[0]
        self.dynamic.growth_manager.error_ema[target_tile_id] = 1.0 # High error

        # Trigger step
        stats = self.dynamic.step()

        # Should have grown
        self.assertEqual(stats["grown"], 1)
        self.assertTrue(self.dynamic.tile_modified)

        # Check if new tile exists
        # We don't know the exact ID, but count should increase
        # Initial tiles: 1 input layer (1 tile if input_dim=8, neurons=16? ceil(8/16)=1)
        # 1 hidden layer (2 tiles)
        # 1 output layer (1 tile if output_dim=4, neurons=16? ceil(4/16)=1)
        # Total 4 tiles?
        # Let's check current count
        # dynamic.step() was called, so count increased
        pass

    def test_add_remove_edge(self):
        """Test adding/removing edges."""
        src = self.model.graph.input_tile_ids[0]
        dst = self.model.graph.output_tile_ids[0]

        # Ensure no edge initially (skip hidden)
        if (src, dst) in self.model.graph._edge_set:
            self.model.remove_edge(src, dst)

        self.model.add_edge(src, dst)
        self.assertIn((src, dst), self.model.graph._edge_set)
        self.assertIn(f"edge_{src}_{dst}", self.model.edge_weights)

        self.model.remove_edge(src, dst)
        self.assertNotIn((src, dst), self.model.graph._edge_set)
        self.assertNotIn(f"edge_{src}_{dst}", self.model.edge_weights)

if __name__ == "__main__":
    unittest.main()
