import unittest

from bioplausible.models.equitile.builder import (
    EnhancedEquiTileBuilder,
    EquiTileBuilder,
)
from bioplausible.models.equitile.enhanced import EnhancedEquiTile
from bioplausible.models.equitile.graph import GraphEquiTile, GraphEquiTileConfig
from bioplausible.models.equitile.timeseries import TimeSeriesConfig, TimeSeriesEquiTile


class TestBuilderCleanup(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_equitile_builder_fluent_methods(self):
        """Test new fluent methods in EquiTileBuilder."""
        model = (
            EquiTileBuilder()
            .with_architecture(32, 2, 2)
            .with_sparsity(threshold=0.5, penalty=0.1)
            .with_importance_learning(lr=0.05, decay=0.9)
            .build()
        )

        config = model.get_config()
        self.assertEqual(config.sparsity_threshold, 0.5)
        self.assertEqual(config.sparsity_penalty_coef, 0.1)
        self.assertEqual(config.importance_lr, 0.05)
        self.assertEqual(config.importance_decay, 0.9)

    def test_enhanced_equitile_builder(self):
        """Test EnhancedEquiTileBuilder works correctly."""
        model = (
            EnhancedEquiTileBuilder()
            .with_architecture(32, 2, 2)
            .with_io(10, 2)
            .enable_layer_norm()
            .with_sparsity(threshold=0.2)
            .build()
        )

        self.assertIsInstance(model, EnhancedEquiTile)

        # In EnhancedEquiTile the specific config object is stored in equitile_config
        config = getattr(model, "equitile_config", None)

        self.assertTrue(getattr(config, "use_layer_norm", False))
        self.assertEqual(getattr(config, "sparsity_threshold", None), 0.2)
        # Check input/output dim
        self.assertEqual(model.W_in.in_features, 10)
        self.assertEqual(model.W_out.out_features, 2)

    def test_graph_equitile_config_cleanup(self):
        """Test GraphEquiTile works without unused fields."""
        config = GraphEquiTileConfig(node_features=5, hidden_dim=16, num_classes=2)
        # Verify removed fields are gone
        self.assertFalse(hasattr(config, "mode"))
        self.assertFalse(hasattr(config, "inference_steps"))

        model = GraphEquiTile(config)
        self.assertIsInstance(model, GraphEquiTile)

    def test_timeseries_equitile_config_cleanup(self):
        """Test TimeSeriesEquiTile works without unused fields."""
        config = TimeSeriesConfig(input_dim=5, seq_len=10, output_dim=1)
        # Verify removed fields are gone
        self.assertFalse(hasattr(config, "mode"))
        self.assertFalse(hasattr(config, "inference_steps"))

        model = TimeSeriesEquiTile(config)
        self.assertIsInstance(model, TimeSeriesEquiTile)


if __name__ == "__main__":
    unittest.main()
