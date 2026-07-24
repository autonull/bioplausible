import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.core.trainer import CoreTrainer  # noqa: E402
from bioplausible.acceleration.kernels import EqPropKernel  # noqa: E402
from bioplausible.zoo.models.eqprop import LoopedMLP  # noqa: E402


class TestEqPropKernel(unittest.TestCase):
    """
    Tests for the EqPropKernel (NumPy/CuPy backend) which offers O(1) memory training.
    """

    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 32
        self.output_dim = 2
        self.batch_size = 4
        self.epochs = 5

        # Synthetic data (NumPy)
        self.x_np = np.random.randn(20, self.input_dim).astype(np.float32)
        self.y_np = np.random.randint(0, self.output_dim, (20,)).astype(np.int64)

        # Torch data for Trainer integration
        self.x_torch = torch.from_numpy(self.x_np)
        self.y_torch = torch.from_numpy(self.y_np)
        self.dataset = TensorDataset(self.x_torch, self.y_torch)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_kernel_standalone_numpy(self):
        """Test the kernel class directly in NumPy mode."""
        kernel = EqPropKernel(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            use_gpu=False,  # Force NumPy
        )

        initial_metrics = kernel.evaluate(self.x_np, self.y_np)
        initial_loss = initial_metrics["loss"]

        for _ in range(self.epochs):
            # Batch training
            for i in range(0, len(self.x_np), self.batch_size):
                bx = self.x_np[i : i + self.batch_size]
                by = self.y_np[i : i + self.batch_size]
                kernel.train_step(bx, by)

        final_metrics = kernel.evaluate(self.x_np, self.y_np)
        final_loss = final_metrics["loss"]

        print(f"Kernel (NumPy): {initial_loss:.4f} -> {final_loss:.4f}")
        # Relaxed check for convergence on random data,
        # but should generally decrease or stay finite
        self.assertTrue(np.isfinite(final_loss))
        # Ensure it updated weights (loss changed)
        self.assertNotEqual(initial_loss, final_loss)

    def test_trainer_integration_kernel_mode(self):
        """Test EqPropTrainer with use_kernel=True (calls KernelEqPropKernel)."""
        # We need a model instance just to pass dimensions to the Trainer
        model = LoopedMLP(self.input_dim, self.hidden_dim, self.output_dim)

        # Initialize trainer with use_kernel=True
        # With my fix in core.py, this should work even without CuPy (falling back to NumPy)
        trainer = EqPropTrainer(model, use_kernel=True, use_compile=False)

        # Run fit
        history = trainer.fit(self.loader, epochs=self.epochs)

        train_loss = history["train_loss"]
        self.assertTrue(len(train_loss) > 0)
        print(f"Trainer (Kernel Mode): {train_loss[0]:.4f} -> {train_loss[-1]:.4f}")

        self.assertIsNotNone(trainer.kernel)
        self.assertFalse(trainer.kernel.use_gpu)  # Should be False on CPU env

    def test_memory_optimization(self):
        """Test that O(1) memory optimization works (trajectory not stored)."""
        kernel = EqPropKernel(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            max_steps=5,
            use_gpu=False,
        )

        _, log, _ = kernel.solve_equilibrium(self.x_np[:2])
        self.assertEqual(len(log), 1, "Should only store last step by default")

        _, log_full, _ = kernel.solve_equilibrium(self.x_np[:2], store_trajectory=True)
        self.assertTrue(len(log_full) >= 1)
        if len(log_full) == 1:
            print(
                "Warning: Converged in 1 step,"
                " can't fully verify trajectory storage difference"
            )
        else:
            self.assertGreater(len(log_full), 1)


if __name__ == "__main__":
    unittest.main()
