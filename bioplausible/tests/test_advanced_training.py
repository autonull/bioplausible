import unittest

import torch

from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.utils import seed_everything
from bioplausible.zoo.models.eqprop import LoopedMLP


class TestAdvancedTraining(unittest.TestCase):
    def setUp(self):
        self.model = LoopedMLP(
            input_dim=10, hidden_dim=20, output_dim=2, use_spectral_norm=False
        )
        self.dataset = [(torch.randn(10), torch.tensor(0)) for _ in range(10)]
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=2)

    def _make_trainer(self, **overrides) -> CoreTrainer:
        config = TrainerConfig(
            model="eqprop_mlp",
            model_kwargs={
                "input_dim": 784,
                "hidden_dim": 20,
                "output_dim": 10,
                "use_spectral_norm": False,
            },
            optimizer="adam",
            optimizer_kwargs={"lr": 1e-3},
            task="mnist",
            epochs=1,
            batch_size=2,
            batches_per_epoch=2,
            val_batches=1,
            grad_clip=0.1,
            **overrides,
        )
        return CoreTrainer(config)

    def test_gradient_clipping(self):
        """Smoke test that CoreTrainer accepts grad_clip."""
        trainer = self._make_trainer()
        history = trainer.fit()
        self.assertIsInstance(history, list)

    def test_amp_smoke(self):
        """Smoke test for AMP path execution."""
        trainer = self._make_trainer(precision="16-mixed")
        history = trainer.fit()
        self.assertIsInstance(history, list)

    def test_seed_everything(self):
        seed_everything(42)
        a = torch.randn(5)

        seed_everything(42)
        b = torch.randn(5)

        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
