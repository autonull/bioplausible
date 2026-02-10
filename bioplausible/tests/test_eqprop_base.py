import unittest

import torch
import torch.nn as nn

from bioplausible.models.eqprop_base import EqPropModel


class MockEqPropModel(EqPropModel):
    """Simple mock implementation of EqPropModel for testing."""

    def __init__(self, input_dim, hidden_dim, output_dim, max_steps=10):
        # We need to set dims before super().__init__ if NEBCBase uses them immediately,
        # or pass them via kwargs. EqPropModel passes kwargs to super.
        self._temp_input = input_dim
        self._temp_hidden = hidden_dim
        self._temp_output = output_dim

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
        )

    def _build_layers(self):
        # Called by super().__init__ -> NEBCBase.__init__
        # At this point, self.input_dim etc are set by NEBCBase
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

    def _initialize_hidden_state(self, x):
        return torch.zeros(x.shape[0], self.hidden_dim, device=x.device)

    def _transform_input(self, x):
        return self.W_in(x)

    def forward_step(self, h, x_transformed):
        return torch.tanh(x_transformed + self.W_rec(h))

    def _output_projection(self, h):
        return self.W_out(h)


class TestEqPropBase(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5
        self.model = MockEqPropModel(self.input_dim, self.hidden_dim, self.output_dim)

    def test_forward_shape(self):
        x = torch.randn(16, self.input_dim)
        out = self.model(x)
        self.assertEqual(out.shape, (16, self.output_dim))

    def test_forward_trajectory(self):
        x = torch.randn(16, self.input_dim)
        steps = 5
        out, traj = self.model(x, steps=steps, return_trajectory=True)
        self.assertEqual(len(traj), steps + 1)  # Initial + 5 steps
        self.assertEqual(traj[0].shape, (16, self.hidden_dim))

    def test_inject_noise_and_relax(self):
        x = torch.randn(16, self.input_dim)
        stats = self.model.inject_noise_and_relax(
            x, noise_level=0.1, injection_step=5, total_steps=10
        )
        self.assertIn("initial_noise", stats)
        self.assertIn("final_noise", stats)
        self.assertIn("damping_ratio", stats)
        self.assertTrue(isinstance(stats["initial_noise"], float))

    def test_compute_lipschitz(self):
        L = self.model.compute_lipschitz()
        self.assertTrue(L >= 0)


if __name__ == "__main__":
    unittest.main()
