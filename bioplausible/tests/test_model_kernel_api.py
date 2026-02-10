import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.core import EqPropTrainer
from bioplausible.kernel import HAS_CUPY
from bioplausible.models.looped_mlp import LoopedMLP


class TestModelKernelAPI(unittest.TestCase):
    """
    Tests for the O(1) Memory API (LoopedMLP with backend='kernel').
    """

    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 32
        self.output_dim = 2
        self.batch_size = 4
        self.epochs = 3

        # Synthetic data
        self.x_np = np.random.randn(20, self.input_dim).astype(np.float32)
        self.y_np = np.random.randint(0, self.output_dim, (20,)).astype(np.int64)

        self.x_torch = torch.from_numpy(self.x_np)
        self.y_torch = torch.from_numpy(self.y_np)
        self.dataset = TensorDataset(self.x_torch, self.y_torch)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_looped_mlp_kernel_backend_init(self):
        """Test initializing LoopedMLP with backend='kernel'."""
        # Force backend='kernel' even if no GPU (falls back to numpy in Kernel class)
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        self.assertEqual(model.backend, "kernel")
        self.assertIsNotNone(model._engine)
        # Check if engine dimensions match
        self.assertEqual(model._engine.input_dim, self.input_dim)

    def test_looped_mlp_forward_kernel(self):
        """Test forward pass via kernel engine."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        x = self.x_torch[:2]
        out = model(x)

        # Output should be a Tensor
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (2, self.output_dim))

    def test_trainer_auto_detection(self):
        """Test that SupervisedTrainer detects the kernel backend."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )

        # Don't pass use_kernel=True explicitly, trainer should infer it
        trainer = EqPropTrainer(model, use_compile=False)

        # Check if trainer adopted use_kernel logic (though self.kernel might be None if model handles it)
        self.assertTrue(trainer.use_kernel)
        self.assertIsNone(
            trainer.kernel
        )  # Should be None because model._engine is used

        # Run fit
        history = trainer.fit(self.loader, epochs=self.epochs)
        train_loss = history["train_loss"]

        self.assertTrue(len(train_loss) > 0)
        print(f"Trainer (Model-Kernel): {train_loss[0]:.4f} -> {train_loss[-1]:.4f}")

    def test_memory_usage_proxy(self):
        """
        Indirectly verify memory usage by checking that gradients are NOT computed
        on the model parameters during training (since kernel handles it).
        """
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )

        trainer = EqPropTrainer(model, use_compile=False)

        # Run one batch
        x, y = next(iter(self.loader))
        trainer.train_batch(x, y)

        # Check PyTorch parameters have no grad
        # LoopedMLP init_weights creates standard nn.Linear layers which have .weight
        # But in kernel mode, these are unused dummies.
        for param in model.parameters():
            self.assertIsNone(param.grad)

    def test_regression_pytorch_backend(self):
        """
        Verify that standard PyTorch backend still works (optimizer exists and updates).
        """
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="pytorch"
        )

        trainer = EqPropTrainer(model, use_compile=False)

        # Verify self.opt exists
        self.assertIsNotNone(trainer.opt)

        # Run one batch
        x, y = next(iter(self.loader))

        # Capture weights before
        w_before = model.W_in.parametrizations.weight.original.clone()

        trainer.train_batch(x, y)

        # Capture weights after
        w_after = model.W_in.parametrizations.weight.original

        # Weights should change
        self.assertFalse(
            torch.allclose(w_before, w_after), "Weights did not update in PyTorch mode"
        )


if __name__ == "__main__":
    unittest.main()
