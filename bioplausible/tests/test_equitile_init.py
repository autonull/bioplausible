import unittest

from bioplausible.equitile.core import EquiTile
from bioplausible.equitile.core import EquiTileConfig


class TestEquiTileInit(unittest.TestCase):
    def test_init_with_kwargs(self):
        """Test initialization with explicit kwargs (legacy mode)."""
        model = EquiTile(
            neurons_per_tile=32,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=5,
            learning_rate=0.05,
            mode="ep",
            task_type="regression",
            activation="relu",
        )
        self.assertEqual(model.equitile_config.neurons_per_tile, 32)
        self.assertEqual(model.equitile_config.learning_rate, 0.05)
        self.assertEqual(model.equitile_config.mode, "ep")
        self.assertEqual(model.task_type, "regression")
        self.assertEqual(model.activation_name, "relu")
        self.assertEqual(model.input_dim, 10)
        self.assertEqual(model.output_dim, 5)

    def test_init_with_config(self):
        """Test initialization with EquiTileConfig object."""
        config = EquiTileConfig(
            neurons_per_tile=16,
            num_layers=4,
            tiles_per_layer=3,
            learning_rate=0.01,
            mode="pc",
            task_type="binary",
            activation="tanh",
        )
        model = EquiTile(config=config, input_dim=20, output_dim=1)
        self.assertEqual(model.equitile_config.neurons_per_tile, 16)
        self.assertEqual(model.equitile_config.mode, "pc")
        self.assertEqual(model.task_type, "binary")
        self.assertEqual(model.activation_name, "tanh")
        self.assertEqual(model.input_dim, 20)
        self.assertEqual(model.output_dim, 1)


if __name__ == "__main__":
    unittest.main()
