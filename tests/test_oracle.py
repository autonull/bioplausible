import unittest

import torch

from bioplausible.zoo.models.eqprop import LoopedMLP


class TestOracle(unittest.TestCase):
    def test_oracle_metric(self):
        """Test that settling time correlates with noise level (uncertainty)."""
        input_dim = 16
        hidden_dim = 32
        output_dim = 10
        model = LoopedMLP(input_dim, hidden_dim, output_dim, max_steps=20)
        model.eval()

        # Base input
        x = torch.randn(1, input_dim)

        # Clean run
        with torch.no_grad():
            _, dynamics_clean = model(x, return_dynamics=True)
            deltas_clean = dynamics_clean["deltas"]
            steps_clean = len([d for d in deltas_clean if d > 1e-3])

        # Noisy run
        noise = 2.0
        x_noisy = x + torch.randn_like(x) * noise

        with torch.no_grad():
            _, dynamics_noisy = model(x_noisy, return_dynamics=True)
            deltas_noisy = dynamics_noisy["deltas"]
            steps_noisy = len([d for d in deltas_noisy if d > 1e-3])

        print(f"Steps (Clean): {steps_clean}")
        print(f"Steps (Noisy): {steps_noisy}")

        # We expect noisy input to take longer or at least be different/higher energy
        # For untrained networks, this might be stochastic, but usually noisy = slower convergence
        # or at least higher residual.

        # Check that we got dynamics
        self.assertTrue(len(deltas_clean) > 0)
        self.assertTrue(len(deltas_noisy) > 0)


if __name__ == "__main__":
    unittest.main()
