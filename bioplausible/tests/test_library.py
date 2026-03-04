"""
Unit tests for eqprop-torch library.
"""

import shutil
# Add parent to path for in-package testing
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible import (ConvEqProp, EqPropTrainer, LoopedMLP,
                          TransformerEqProp, compile_model, count_parameters,
                          create_model_preset, verify_spectral_norm)
from bioplausible.acceleration import enable_tf32


class TestModels(unittest.TestCase):
    """Test core models."""

    def test_looped_mlp_creation(self):
        """Test LoopedMLP initialization and forward pass."""
        model = LoopedMLP(784, 256, 10, use_spectral_norm=True)
        self.assertEqual(count_parameters(model) > 0, True)

        # Check spectral norm
        L = model.compute_lipschitz()

        # It's okay if it's slightly > 1.0 due to power iteration error
        self.assertLess(L, 1.1)

        # Forward pass
        x = torch.randn(2, 784)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_conv_eqprop(self):
        """Test ConvEqProp."""
        model = ConvEqProp(1, 32, 10)
        x = torch.randn(2, 1, 28, 28)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_transformer_eqprop(self):
        """Test TransformerEqProp."""
        model = TransformerEqProp(
            vocab_size=100,
            hidden_dim=32,
            output_dim=100,  # Required: normally same as vocab_size for LM
            num_layers=2,
            max_seq_len=64,
        )
        x = torch.randint(0, 100, (2, 64))
        y = model(x)
        # Model returns classification/prediction for the sequence (B, output_dim)
        self.assertEqual(y.shape, (2, 100))


class TestTrainer(unittest.TestCase):
    """Test EqPropTrainer."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create dummy tasks
        self.model = LoopedMLP(50, 32, 5, use_spectral_norm=True).to(self.device)

        # Dummy data
        self.x = torch.randn(32, 50)
        self.y = torch.randint(0, 5, (32,))
        self.dataset = TensorDataset(self.x, self.y)
        self.loader = DataLoader(self.dataset, batch_size=8)

        self.checkpoint_dir = Path("test_checkpoints")
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir()

    def tearDown(self):
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)

    def test_init_validation(self):
        """Test constructor validation."""
        # Invalid optimizer
        # with self.assertRaises(ValueError):
        #    EqPropTrainer(self.model, optimizer="invalid_opt")

        # Invalid compile mode
        with self.assertRaises(ValueError):
            EqPropTrainer(self.model, compile_mode="invalid_mode")

        # Invalid learning rate
        with self.assertRaises(ValueError):
            EqPropTrainer(self.model, lr=-0.1)

    def test_fit_simple(self):
        """Test simple training loop (cpu/cuda)."""
        trainer = EqPropTrainer(self.model, use_compile=False, device=self.device)
        history = trainer.fit(self.loader, epochs=2)

        self.assertTrue("train_loss" in history)
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(trainer.current_epoch, 1)  # 0-indexed, so 2 epochs end at 1

    def test_checkpointing(self):
        """Test save/load checkpoint."""
        trainer = EqPropTrainer(self.model, use_compile=False, device=self.device)
        path = self.checkpoint_dir / "ckpt.pt"

        trainer.save_checkpoint(str(path))
        self.assertTrue(path.exists())

        # Load back
        trainer.load_checkpoint(str(path))

    def test_tf32_config(self):
        """Test TF32 configuration."""
        # Just ensure it doesn't crash
        trainer = EqPropTrainer(self.model, use_compile=False, allow_tf32=True)
        self.assertTrue(trainer)


class TestUtils(unittest.TestCase):
    """Test utilities."""

    def test_presets(self):
        model = create_model_preset("mnist_small")
        self.assertIsInstance(model, LoopedMLP)

        with self.assertRaises(ValueError):
            create_model_preset("invalid_preset")

    def test_verify_spectral_norm(self):
        model = LoopedMLP(50, 32, 5, use_spectral_norm=True)
        sn_vals = verify_spectral_norm(model)
        self.assertTrue(len(sn_vals) > 0)


if __name__ == "__main__":
    unittest.main()
