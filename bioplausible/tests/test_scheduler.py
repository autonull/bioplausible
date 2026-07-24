import unittest

import torch
from torch.optim.lr_scheduler import StepLR

from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.zoo.models.eqprop import LoopedMLP


class TestSchedulerIntegration(unittest.TestCase):
    def setUp(self):
        self.model = LoopedMLP(
            input_dim=784, hidden_dim=20, output_dim=10, use_spectral_norm=False
        )
        config = TrainerConfig(
            model="eqprop_mlp",
            model_kwargs={
                "input_dim": 784,
                "hidden_dim": 20,
                "output_dim": 10,
                "use_spectral_norm": False,
            },
            optimizer="adam",
            task="mnist",
            epochs=1,
            batch_size=5,
            batches_per_epoch=2,
            val_batches=1,
            use_compile=False,
        )
        self.trainer = CoreTrainer(config)
        self.trainer.setup()

    def test_scheduler_step(self):
        """Verify a StepLR scheduler steps at end of each epoch."""
        initial_lr = 0.1
        optimizer = torch.optim.SGD(self.model.parameters(), lr=initial_lr)
        self.trainer.optimizer = optimizer

        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        self.trainer.fit(scheduler=scheduler)
        expected_lr = initial_lr * 0.1
        current_lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(current_lr, expected_lr, places=5)

        self.trainer.fit(scheduler=scheduler)
        expected_lr = initial_lr * 0.01
        current_lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(current_lr, expected_lr, places=5)

    def test_scheduler_kernel_warning(self):
        """When the model uses the kernel backend, scheduler step logs a warning."""
        # Reconfigure model with kernel backend.
        self.trainer.model = LoopedMLP(
            input_dim=784, hidden_dim=20, output_dim=10, backend="kernel"
        )
        self.trainer.config.batches_per_epoch = 1
        self.trainer.config.val_batches = 1

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        with self.assertLogs(
            logger="bioplausible.core.trainer", level="WARNING"
        ) as cm:
            self.trainer.fit(scheduler=scheduler)
        self.assertTrue(
            any("kernel" in msg.lower() for msg in cm.output),
            f"Expected kernel warning in logs: {cm.output}",
        )


if __name__ == "__main__":
    unittest.main()
