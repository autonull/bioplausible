import unittest

import torch

from bioplausible.models.equitile.config import EquiTileConfig
from bioplausible.models.equitile.core import EquiTile
from bioplausible.models.equitile.language import LMEquiTile, LMEquiTileConfig
from bioplausible.models.equitile.rl import RLEquiTile, RLEquiTileConfig
from bioplausible.models.equitile.vision import (ConvEquiTile,
                                                 ConvEquiTileConfig)


class TestEquiTileCleanup(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_vision_kwargs(self):
        """Test passing kwargs to ConvEquiTile."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            equitile_kwargs={"sparsity_threshold": 0.5},
        )
        model = ConvEquiTile(config)
        self.assertEqual(model.head.get_config().sparsity_threshold, 0.5)

    def test_lm_kwargs(self):
        """Test passing kwargs to LMEquiTile."""
        config = LMEquiTileConfig(
            vocab_size=100,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            max_seq_len=20,
            equitile_kwargs={"importance_lr": 0.05},
        )
        model = LMEquiTile(config)
        # Check first layer
        self.assertEqual(model.layers[0].equitile.get_config().importance_lr, 0.05)

    def test_rl_kwargs(self):
        """Test passing kwargs to RLEquiTile."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            equitile_kwargs={"dropout": 0.3},
        )
        model = RLEquiTile(config)
        self.assertEqual(model.feature_extractor.get_config().dropout, 0.3)

    def test_core_get_config(self):
        """Test get_config on Core EquiTile."""
        model = EquiTile(neurons_per_tile=16)
        config = model.get_config()
        self.assertIsInstance(config, EquiTileConfig)
        self.assertEqual(config.neurons_per_tile, 16)

    def test_input_output_dim_consistency(self):
        """Test that input_dim and output_dim are consistent."""
        model = EquiTile(input_dim=12, output_dim=6)
        self.assertEqual(model.input_dim, 12)
        self.assertEqual(model.output_dim, 6)
        self.assertEqual(model.config.input_dim, 12)
        self.assertEqual(model.config.output_dim, 6)


if __name__ == "__main__":
    unittest.main()
