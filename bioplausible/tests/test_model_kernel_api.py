import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.zoo.models.eqprop import LoopedMLP  # noqa: E402


class TestModelKernelAPI(unittest.TestCase):
    """Tests for the O(1) Memory API (LoopedMLP with backend='kernel')."""

    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 32
        self.output_dim = 2
        self.batch_size = 4

        self.x_np = np.random.randn(20, self.input_dim).astype(np.float32)
        self.y_np = np.random.randint(0, self.output_dim, (20,)).astype(np.int64)

        self.x_torch = torch.from_numpy(self.x_np)
        self.y_torch = torch.from_numpy(self.y_np)
        self.dataset = TensorDataset(self.x_torch, self.y_torch)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_looped_mlp_kernel_backend_init(self):
        """Test initializing LoopedMLP with backend='kernel'."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        self.assertEqual(model.backend, "kernel")
        self.assertIsNotNone(model._engine)
        self.assertEqual(model._engine.input_dim, self.input_dim)

    def test_looped_mlp_forward_kernel(self):
        """Test forward pass via kernel engine."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        x = self.x_torch[:2]
        out = model(x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2, self.output_dim))

    def test_model_kernel_forward_no_grad(self):
        """In kernel mode, the LoopedMLP forward path goes through the engine,
        leaving PyTorch parameters' .grad unset until external training.
        """
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        out = model(self.x_torch[:2])
        # Forward only; no backward called, so grads remain None.
        for param in model.parameters():
            self.assertIsNone(param.grad)
        self.assertIsInstance(out, torch.Tensor)

    def test_regression_pytorch_backend(self):
        """Verify that the standard PyTorch backend still updates weights."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="pytorch"
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()

        x, y = next(iter(self.loader))
        w_before = model.W_in.parametrizations.weight.original.clone()

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        w_after = model.W_in.parametrizations.weight.original
        self.assertFalse(
            torch.allclose(w_before, w_after),
            "Weights did not update in PyTorch mode",
        )


if __name__ == "__main__":
    unittest.main()
