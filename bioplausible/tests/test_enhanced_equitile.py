import unittest
import torch
from bioplausible.models.equitile.enhanced import EnhancedEquiTile, EnhancedEquiTileConfig

class TestEnhancedEquiTile(unittest.TestCase):
    def test_initialization(self):
        """Test basic initialization."""
        model = EnhancedEquiTile(
            neurons_per_tile=32,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=5,
            use_layer_norm=True
        )
        self.assertIsInstance(model, EnhancedEquiTile)
        self.assertEqual(model.config.neurons_per_tile, 32)

    def test_activation(self):
        """Test activation module usage."""
        model = EnhancedEquiTile(
            neurons_per_tile=16,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=5,
            activation="relu"
        )
        self.assertIsInstance(model.activation, torch.nn.ReLU)

    def test_metrics(self):
        """Test compute_metrics implementation."""
        model = EnhancedEquiTile(
            neurons_per_tile=16,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=5,
            task_type="classification"
        )

        logits = torch.randn(4, 5)
        # Perfect prediction
        y = logits.argmax(dim=-1)

        acc = model.compute_metrics(logits, y)
        self.assertEqual(acc, 1.0)

    def test_train_step(self):
        """Test training step with enhancements."""
        model = EnhancedEquiTile(
            neurons_per_tile=16,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=2,
            use_layer_norm=True,
            use_curriculum=False
        )

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        stats = model.train_step(x, y)
        self.assertIn("loss", stats)
        self.assertIn("accuracy", stats)
        self.assertIn("mean_error", stats)

if __name__ == "__main__":
    unittest.main()
