import unittest

import torch

from bioplausible.hyperopt.tasks import VisionTask, create_task
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec


class TestVisionTask(unittest.TestCase):
    def _test_task(self, task_name, expected_channels, expected_h, expected_w):
        """Helper to test VisionTask setup and training loop."""
        print(f"Testing task: {task_name}")
        task = create_task(task_name, device="cpu", quick_mode=True)
        self.assertIsInstance(task, VisionTask, f"{task_name} should be VisionTask")

        task.setup()

        # Verify shape (N, C, H, W)
        x, y = task.get_batch(split="train", batch_size=4)
        self.assertEqual(x.dim(), 4, f"{task_name}: Expected 4D input (NCHW)")
        self.assertEqual(
            x.shape[1],
            expected_channels,
            f"{task_name}: Expected {expected_channels} channel(s)",
        )
        self.assertEqual(
            x.shape[2], expected_h, f"{task_name}: Expected height {expected_h}"
        )
        self.assertEqual(
            x.shape[3], expected_w, f"{task_name}: Expected width {expected_w}"
        )

        # Create compatible model
        spec = get_model_spec("Backprop Baseline")
        model = create_model(
            spec,
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            hidden_dim=32,
            num_layers=1,
            task_type="vision",
        )

        trainer = task.create_trainer(model)

        # Run one epoch
        metrics = trainer.train_epoch()
        self.assertIn("loss", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_digits(self):
        self._test_task("digits", 1, 8, 8)

    def test_usps(self):
        # USPS is 16x16 grayscale
        self._test_task("usps", 1, 16, 16)

    def test_kmnist(self):
        self._test_task("kmnist", 1, 28, 28)

    def test_fashion_mnist(self):
        self._test_task("fashion_mnist", 1, 28, 28)

    def test_cifar10(self):
        self._test_task("cifar10", 3, 32, 32)

    def test_cifar100(self):
        self._test_task("cifar100", 3, 32, 32)

    def test_svhn(self):
        self._test_task("svhn", 3, 32, 32)


if __name__ == "__main__":
    unittest.main()
