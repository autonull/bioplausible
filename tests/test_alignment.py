import copy
import unittest

import torch
import torch.nn as nn

from bioplausible.models.looped_mlp import LoopedMLP


class TestAlignment(unittest.TestCase):
    def test_alignment_calculation(self):
        """Test gradient alignment logic."""
        # 1. Create model (BPTT mode for perfect alignment test)
        input_dim = 16
        hidden_dim = 32
        output_dim = 10

        # LoopedMLP default uses BPTT if gradient_method is not specified
        model = LoopedMLP(
            input_dim, hidden_dim, output_dim, max_steps=5, gradient_method="bptt"
        )

        # Data
        x = torch.randn(4, input_dim)
        y = torch.randint(0, output_dim, (4,))

        # 2. Get BP Gradients
        model_bp = copy.deepcopy(model)
        model_bp.zero_grad()
        out = model_bp(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        grads_bp = [
            p.grad.flatten() for p in model_bp.parameters() if p.grad is not None
        ]

        # 3. Simulate "EqProp" update (using same BPTT logic to verify 1.0 alignment)
        # In the worker, we force an update and measure delta.
        # Let's verify that delta_W via SGD(lr=1) matches -grad.

        model_eq = copy.deepcopy(model)
        # Mock optimizer
        optimizer = torch.optim.SGD(model_eq.parameters(), lr=1.0)
        optimizer.zero_grad()
        out_eq = model_eq(x)
        loss_eq = nn.functional.cross_entropy(out_eq, y)
        loss_eq.backward()
        optimizer.step()

        # Measure delta
        grads_eq = []
        for (n_bp, p_bp), (n_eq, p_eq) in zip(
            model_bp.named_parameters(), model_eq.named_parameters()
        ):
            if p_bp.grad is not None:
                delta = p_bp.data - p_eq.data  # w_old - w_new = lr * grad = 1.0 * grad
                grads_eq.append(delta.flatten())

        # 4. Compare
        self.assertTrue(len(grads_bp) > 0)
        self.assertEqual(len(grads_bp), len(grads_eq))

        for g_bp, g_eq in zip(grads_bp, grads_eq):
            sim = torch.nn.functional.cosine_similarity(
                g_bp.unsqueeze(0), g_eq.unsqueeze(0)
            ).item()
            print(f"Alignment: {sim}")
            # Should be very close to 1.0
            self.assertGreater(sim, 0.999)


if __name__ == "__main__":
    unittest.main()
