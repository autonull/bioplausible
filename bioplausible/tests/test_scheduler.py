import unittest

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from bioplausible.core import EqPropTrainer
from bioplausible.models.looped_mlp import LoopedMLP


class TestSchedulerIntegration(unittest.TestCase):
    def setUp(self):
        self.model = LoopedMLP(
            input_dim=10, hidden_dim=20, output_dim=2, use_spectral_norm=False
        )
        self.trainer = EqPropTrainer(self.model, use_compile=False)
        self.dataset = [(torch.randn(10), torch.tensor(0)) for _ in range(20)]
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=5)

    def test_scheduler_step(self):
        # Initial LR
        initial_lr = 0.1
        optimizer = torch.optim.SGD(self.model.parameters(), lr=initial_lr)
        self.trainer.optimizer = optimizer  # Override default optimizer

        # Scheduler: decay by 0.1 every epoch
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        # Run 1 epoch
        self.trainer.fit(self.loader, epochs=1, scheduler=scheduler)

        # Check LR after 1 epoch
        # StepLR steps at the end of epoch, so after 1 epoch, LR should be initial_lr * 0.1
        expected_lr = initial_lr * 0.1
        current_lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(current_lr, expected_lr, places=5)

        # Run another epoch
        self.trainer.fit(self.loader, epochs=1, scheduler=scheduler)
        expected_lr = initial_lr * 0.01
        current_lr = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(current_lr, expected_lr, places=5)

    def test_scheduler_kernel_warning(self):
        # If kernel mode is used, scheduler should warn
        self.trainer.use_kernel = True

        # Mock kernel to avoid errors
        class MockKernel:
            def train_step(self, x, y):
                return {"loss": 0.1, "accuracy": 0.9}

            def evaluate(self, x, y):
                return {"loss": 0.1, "accuracy": 0.9}

        self.trainer.kernel = MockKernel()
        self.trainer.model.input_dim = 10
        self.trainer.model.hidden_dim = 20
        self.trainer.model.output_dim = 2

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        with self.assertWarns(UserWarning):
            self.trainer.fit(self.loader, epochs=1, scheduler=scheduler)


if __name__ == "__main__":
    unittest.main()
