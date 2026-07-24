import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from bioplausible import EqPropTrainer, LoopedMLP


class TestCoreTrainer(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5
        self.batch_size = 4
        self.model = LoopedMLP(self.input_dim, self.hidden_dim, self.output_dim)

        # Create dummy data
        x = torch.randn(16, self.input_dim)
        y = torch.randint(0, self.output_dim, (16,))
        self.dataset = TensorDataset(x, y)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size)

    def test_fit_and_evaluate(self):
        trainer = EqPropTrainer(self.model, use_compile=False, device="cpu")

        # Test fit
        history = trainer.fit(self.loader, epochs=1, val_loader=self.loader)
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertTrue(len(history["train_loss"]) == 1)

        # Test evaluate explicit call
        metrics = trainer.evaluate(self.loader)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(metrics["loss"], float)
        self.assertIsInstance(metrics["accuracy"], float)


if __name__ == "__main__":
    unittest.main()
