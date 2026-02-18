import time
import unittest

import torch

from bioplausible.models.base import ModelConfig
from bioplausible.models.finite_nudge_ep import FiniteNudgeEP


class TestFiniteNudge(unittest.TestCase):
    def test_finite_nudge_execution(self):
        """
        Verify that FiniteNudgeEP runs without hanging (on CPU).
        """
        config = ModelConfig(
            name="FiniteNudge",
            input_dim=10,
            hidden_dims=[20],
            output_dim=2,
            learning_rate=0.01,
            beta=1.0,
            equilibrium_steps=5,
        )
        model = FiniteNudgeEP(config)

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        start_time = time.time()
        # First run (might trigger compile if not disabled)
        metrics = model.train_step(x, y)
        duration = time.time() - start_time

        print(f"First step duration: {duration:.4f}s")
        self.assertLess(
            duration, 5.0, "First step took too long (likely stuck in compilation)"
        )

        # Second run
        start_time = time.time()
        metrics = model.train_step(x, y)
        duration = time.time() - start_time
        print(f"Second step duration: {duration:.4f}s")

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
