import unittest

import torch
import torch.nn as nn

from bioplausible import EqPropTrainer
from bioplausible.models import AdaptiveFA


class TestAdaptiveFA(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AdaptiveFA(
            input_dim=10, hidden_dim=20, output_dim=5, num_layers=2
        ).to(self.device)
        self.trainer = EqPropTrainer(
            self.model, use_compile=False
        )  # Disable compile for simple test

    def test_forward(self):
        x = torch.randn(4, 10, device=self.device)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_training_step(self):
        x = torch.randn(4, 10, device=self.device)
        y = torch.randint(0, 5, (4,), device=self.device)

        # Capture weights before
        w_before = self.model.layers[0].weight.data.clone()
        b_before = self.model.feedback_weights[1].data.clone()

        metrics = self.model.train_step(x, y)

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)

        # Check if weights updated
        w_after = self.model.layers[0].weight.data
        self.assertFalse(torch.allclose(w_before, w_after))

        # Check if feedback weights updated (Alignment)
        b_after = self.model.feedback_weights[1].data
        self.assertFalse(torch.allclose(b_before, b_after))


if __name__ == "__main__":
    unittest.main()
