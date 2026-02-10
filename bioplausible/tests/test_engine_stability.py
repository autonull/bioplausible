import unittest

import torch
import torch.nn as nn

from bioplausible.models.looped_mlp import LoopedMLP


class TestEngineStability(unittest.TestCase):
    def test_equilibrium_stability(self):
        """
        Verify that gradient_method='equilibrium' runs for many iterations
        without crashing (graph retention error) or diverging instantly.
        This addresses the concern in the former NEXT_STEPS.md.
        """
        print("\nTesting Equilibrium Engine Stability...")

        # Setup
        input_dim = 64
        hidden_dim = 64
        output_dim = 10
        batch_size = 32
        steps = 10

        model = LoopedMLP(
            input_dim,
            hidden_dim,
            output_dim,
            max_steps=steps,
            gradient_method="equilibrium",
            use_spectral_norm=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(batch_size, input_dim)
        y = torch.randint(0, output_dim, (batch_size,))

        # Train for 50 iterations (should trigger graph errors if any exist)
        model.train()
        initial_loss = 0.0

        for i in range(50):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = loss.item()

            if i % 10 == 0:
                print(f"  Iter {i}: Loss {loss.item():.4f}")

        final_loss = loss.item()
        print(f"  Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")

        # Assertions
        self.assertLess(
            final_loss, initial_loss, "Loss should decrease (learning should happen)"
        )
        # If we reached here without RuntimeError, the graph retention is likely correct.


if __name__ == "__main__":
    unittest.main()
