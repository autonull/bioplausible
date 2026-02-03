import unittest

import torch

from bioplausible.models.looped_mlp import LoopedMLP


class TestDreaming(unittest.TestCase):
    def test_dreaming_optimization(self):
        """Test that optimizing input x works on an EqPropModel."""
        input_dim = 16
        hidden_dim = 32
        output_dim = 10
        model = LoopedMLP(input_dim, hidden_dim, output_dim, max_steps=5)
        model.eval()  # Dreaming usually done in eval mode

        # Random input
        x = torch.randn(1, input_dim, requires_grad=True)
        target_class = 5

        # Initial score
        with torch.no_grad():
            initial_score = model(x)[0, target_class].item()

        optimizer = torch.optim.SGD([x], lr=0.1)

        # Optimize for 10 steps
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = -out[0, target_class]
            loss.backward()
            optimizer.step()

        # Final score
        with torch.no_grad():
            final_score = model(x)[0, target_class].item()

        # Assert score increased
        print(f"Initial: {initial_score}, Final: {final_score}")
        self.assertGreater(final_score, initial_score)

        # Assert x changed
        self.assertTrue(x.grad is not None)


if __name__ == "__main__":
    unittest.main()
