"""
Tests for robustness and edge cases.
"""

import sys
import unittest
from pathlib import Path

import torch

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from torch.utils.data import DataLoader, TensorDataset

from bioplausible import EqPropTrainer, LoopedMLP


class TestRobustness(unittest.TestCase):
    """Test robustness and edge cases."""

    def test_looped_mlp_dimension_mismatch(self):
        """Test LoopedMLP raises error on dimension mismatch."""
        model = LoopedMLP(10, 5, 2)
        x = torch.randn(4, 11)  # Wrong input dim (11 vs 10)

        with self.assertRaises(ValueError) as cm:
            model(x)
        self.assertIn("Input dimension mismatch", str(cm.exception))

    def test_trainer_invalid_loader(self):
        """Test trainer raises error for non-iterable loader."""
        model = LoopedMLP(10, 5, 2)
        trainer = EqPropTrainer(model, use_compile=False)

        with self.assertRaises(ValueError) as cm:
            trainer.fit(train_loader="not_iterable")

        self.assertIn("must be an iterable DataLoader", str(cm.exception))

    def test_trainer_empty_dataset(self):
        """Test trainer handles empty dataset."""
        model = LoopedMLP(10, 5, 2)
        trainer = EqPropTrainer(model, use_compile=False)

        # Create an empty dataset
        empty_dataset = TensorDataset(torch.empty(0, 10), torch.empty(0))
        empty_loader = DataLoader(empty_dataset, batch_size=2)

        # Should not crash, just do 0 updates
        history = trainer.fit(empty_loader, epochs=1)
        self.assertEqual(len(history["train_loss"]), 1)
        self.assertEqual(trainer.current_epoch, 0)  # 0-indexed


if __name__ == "__main__":
    unittest.main()
