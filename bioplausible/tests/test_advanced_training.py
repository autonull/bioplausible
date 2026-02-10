import unittest

import torch
import torch.nn as nn

from bioplausible.core import EqPropTrainer
from bioplausible.models.looped_mlp import LoopedMLP
from bioplausible.utils import seed_everything


class TestAdvancedTraining(unittest.TestCase):
    def setUp(self):
        # Small model for quick tests
        self.model = LoopedMLP(
            input_dim=10, hidden_dim=20, output_dim=2, use_spectral_norm=False
        )
        self.dataset = [(torch.randn(10), torch.tensor(0)) for _ in range(10)]
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=2)

    def test_gradient_clipping(self):
        # Smoke test for gradient clipping
        trainer = EqPropTrainer(self.model, use_compile=False)
        # We can't easily assert gradients were clipped without hooks, but we ensure it runs
        history = trainer.fit(self.loader, epochs=1, max_grad_norm=0.1)
        self.assertIn("train_loss", history)

    def test_amp_smoke(self):
        # Smoke test for AMP.
        # On CPU, torch.amp.autocast works with bfloat16/float16 but might be slow or no-op depending on hardware.
        # We just want to ensure the code path executes without crashing.
        trainer = EqPropTrainer(self.model, use_compile=False, use_amp=True)
        history = trainer.fit(self.loader, epochs=1)
        self.assertIn("train_loss", history)

    def test_seed_everything(self):
        seed_everything(42)
        a = torch.randn(5)

        seed_everything(42)
        b = torch.randn(5)

        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
