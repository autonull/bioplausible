import unittest

import torch

from bioplausible.models.equitile.core import EquiTile
from bioplausible.models.equitile.language import LMEquiTile
from bioplausible.models.equitile.rl import RLEquiTile
from bioplausible.models.equitile.vision import ConvEquiTile


class TestEquiTileRefactor(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_equitile_core_backprop(self):
        """Test EquiTile in backprop mode."""
        model = EquiTile(
            neurons_per_tile=16,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=5,
            mode="backprop",
        ).to(self.device)
        x = torch.randn(2, 10).to(self.device)
        y = torch.randint(0, 5, (2,)).to(self.device)

        # Forward
        logits = model(x)
        self.assertEqual(logits.shape, (2, 5))

        # Train step
        stats = model.train_step(x, y)
        self.assertIn("loss", stats)
        self.assertIn("accuracy", stats)

    def test_equitile_core_pc(self):
        """Test EquiTile in PC mode."""
        model = EquiTile(
            neurons_per_tile=16,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=5,
            mode="pc",
            inference_steps=2,
        ).to(self.device)
        x = torch.randn(2, 10).to(self.device)
        y = torch.randint(0, 5, (2,)).to(self.device)

        # Forward
        logits = model(x)
        self.assertEqual(logits.shape, (2, 5))

        # Train step
        stats = model.train_step(x, y)
        self.assertIn("loss", stats)
        self.assertIn("active_tiles", stats)

    def test_equitile_core_ep(self):
        """Test EquiTile in EP mode."""
        model = EquiTile(
            neurons_per_tile=16,
            num_layers=3,
            tiles_per_layer=2,
            input_dim=10,
            output_dim=5,
            mode="ep",
            inference_steps=2,
        ).to(self.device)
        x = torch.randn(2, 10).to(self.device)
        y = torch.randint(0, 5, (2,)).to(self.device)

        # Forward
        logits = model(x)
        self.assertEqual(logits.shape, (2, 5))

        # Train step
        stats = model.train_step(x, y)
        self.assertIn("loss", stats)
        self.assertIn("beta", stats)

    def test_conv_equitile(self):
        """Test ConvEquiTile."""
        model = ConvEquiTile(
            input_channels=3,
            input_size=32,
            num_classes=10,
            neurons_per_tile=16,
            tiles_per_layer=2,
            conv_channels=[4, 8],
            mode="pc",
        ).to(self.device)
        x = torch.randn(2, 3, 32, 32).to(self.device)
        y = torch.randint(0, 10, (2,)).to(self.device)

        # Forward
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))

        # Train step
        stats = model.train_step(x, y)
        self.assertIn("loss", stats)

    def test_lm_equitile(self):
        """Test LMEquiTile."""
        vocab_size = 50
        model = LMEquiTile(
            vocab_size=vocab_size,
            embed_dim=16,
            num_heads=2,
            num_layers=2,
            max_seq_len=20,
            neurons_per_tile=16,
            tiles_per_layer=2,
        ).to(self.device)
        x = torch.randint(0, vocab_size, (2, 10)).to(self.device)

        # Forward
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10, vocab_size))

        # Generate
        gen = model.generate(x[:, :5], max_length=10)
        self.assertEqual(gen.shape, (2, 10))

    def test_rl_equitile(self):
        """Test RLEquiTile."""
        model = RLEquiTile(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
            neurons_per_tile=16,
            tiles_per_layer=2,
        ).to(self.device)
        obs = torch.randn(2, 8).to(self.device)

        # Act
        action, value, log_prob = model.act(obs)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(value.shape, (2,))
        self.assertEqual(log_prob.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
