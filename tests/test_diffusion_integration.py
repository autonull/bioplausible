import unittest

import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo.models.eqprop import EqPropDiffusion


class TestDiffusionIntegration(unittest.TestCase):
    def test_factory_creation(self):
        """Test that the factory can create the diffusion model."""
        model_cls = Registry.get(ComponentCategory.MODEL, "eqprop_diffusion")
        model = model_cls(img_channels=1, hidden_channels=32)
        self.assertIsInstance(model, EqPropDiffusion)
        self.assertEqual(model.img_channels, 1)
        # Check hidden dim of underlying denoiser if possible
        self.assertEqual(model.denoiser.hidden_channels, 32)

    def test_train_step(self):
        """Test a single training step."""
        model = EqPropDiffusion(img_channels=1, hidden_channels=32)
        # Input: [B, C, H, W]
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))  # Labels (ignored)

        metrics = model.train_step(x, y)
        self.assertIn("loss", metrics)
        # Loss should be float or tensor
        self.assertTrue(
            isinstance(metrics["loss"], float)
            or isinstance(metrics["loss"], torch.Tensor)
        )

    def test_sample(self):
        """Test sampling."""
        model = EqPropDiffusion(img_channels=1, hidden_channels=32)
        model.eval()

        # Test small sample
        with torch.no_grad():
            samples = model.sample(num_samples=2, img_size=(1, 16, 16), device="cpu")

        self.assertEqual(samples.shape, (2, 1, 16, 16))
        # Check range roughly
        self.assertTrue(samples.max() <= 1.0)
        self.assertTrue(samples.min() >= -1.0)


if __name__ == "__main__":
    unittest.main()
