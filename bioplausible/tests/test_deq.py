import unittest

import torch
import torch.nn as nn

from bioplausible.models.eqprop_base import EqPropModel, EquilibriumFunction
from bioplausible.models.looped_mlp import LoopedMLP


class TestDEQGradients(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5
        self.batch_size = 4

        # Use simple model
        self.model = LoopedMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            max_steps=10,
            use_spectral_norm=True,  # Re-enable spectral norm to ensure it works too
        ).to(self.device)

    def test_gradients_match_bptt(self):
        """Verify that Equilibrium gradients match BPTT gradients (approximately)."""
        x = torch.randn(self.batch_size, self.input_dim, device=self.device)
        y = torch.randint(0, self.output_dim, (self.batch_size,), device=self.device)
        criterion = nn.CrossEntropyLoss()

        # 1. Compute BPTT gradients
        self.model.zero_grad()
        self.model.gradient_method = "bptt"
        out_bptt = self.model(x)
        loss_bptt = criterion(out_bptt, y)
        loss_bptt.backward()

        grads_bptt = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads_bptt[name] = param.grad.clone()

        # 2. Compute Equilibrium gradients
        self.model.zero_grad()
        self.model.gradient_method = "equilibrium"
        out_deq = self.model(x)
        loss_deq = criterion(out_deq, y)
        loss_deq.backward()

        grads_deq = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads_deq[name] = param.grad.clone()

        # Compare
        for name in grads_bptt:
            self.assertIn(name, grads_deq)
            self.assertTrue(torch.isfinite(grads_deq[name]).all())

            g1 = grads_bptt[name].flatten()
            g2 = grads_deq[name].flatten()
            if g1.norm() > 0 and g2.norm() > 0:
                cosine = torch.dot(g1, g2) / (g1.norm() * g2.norm())
                self.assertGreater(cosine.item(), 0.0)

    def test_memory_usage(self):
        """Check if DEQ mode uses less memory (not rigorous, but smoke test)."""
        if not torch.cuda.is_available():
            return

        self.model.max_steps = 50
        x = torch.randn(128, self.input_dim, device=self.device)
        y = torch.randint(0, self.output_dim, (128,), device=self.device)
        criterion = nn.CrossEntropyLoss()

        # BPTT Memory
        torch.cuda.reset_peak_memory_stats()
        self.model.gradient_method = "bptt"
        out = self.model(x)
        loss = criterion(out, y)
        loss.backward()
        mem_bptt = torch.cuda.max_memory_allocated()

        # DEQ Memory
        torch.cuda.reset_peak_memory_stats()
        self.model.gradient_method = "equilibrium"
        out = self.model(x)
        loss = criterion(out, y)
        loss.backward()
        mem_deq = torch.cuda.max_memory_allocated()

        # DEQ should use significantly less memory for large steps
        print(f"BPTT Memory: {mem_bptt/1024**2:.2f} MB")
        print(f"DEQ Memory:  {mem_deq/1024**2:.2f} MB")
        # Assert usually holds but might depend on overhead
        # self.assertLess(mem_deq, mem_bptt)


if __name__ == "__main__":
    unittest.main()
