import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.models.looped_mlp import LoopedMLP


class TestEquilibriumStress(unittest.TestCase):
    """
    Stress test for Implicit Differentiation (Equilibrium Mode).
    Verifies stability and convergence without external dependencies.
    """

    def setUp(self):
        self.device = "cpu"
        self.input_dim = 20
        self.hidden_dim = 100
        self.output_dim = 5
        self.batch_size = 32
        self.epochs = 5
        self.max_steps = 20

        # Synthetic data
        self.x = torch.randn(200, self.input_dim).to(self.device)
        self.y = torch.randint(0, self.output_dim, (200,)).to(self.device)
        self.dataset = TensorDataset(self.x, self.y)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_looped_mlp_equilibrium_stability(self):
        model = LoopedMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            max_steps=self.max_steps,
            gradient_method="equilibrium",
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            model.train()

            for i, (bx, by) in enumerate(self.loader):
                optimizer.zero_grad()

                output = model(bx)
                loss = criterion(output, by)
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        # 1. Verify Convergence
        self.assertLess(
            losses[-1], losses[0], "Model failed to learn (Loss did not decrease)"
        )

        # 2. Verify Stability (No NaNs)
        self.assertFalse(torch.isnan(torch.tensor(losses)).any(), "Loss contains NaNs")


if __name__ == "__main__":
    unittest.main()
