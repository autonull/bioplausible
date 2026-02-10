import unittest

import torch
import torch.nn as nn

from bioplausible.models.conv_eqprop import ConvEqProp
from bioplausible.models.looped_mlp import LoopedMLP


class TestEquilibriumParity(unittest.TestCase):
    def test_mlp_gradient_parity(self):
        print("\nTesting MLP Gradient Parity (BPTT vs Equilibrium)...")
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        batch_size = 4
        max_steps = 100

        torch.manual_seed(42)

        x = torch.randn(batch_size, input_dim)
        y = torch.randint(0, output_dim, (batch_size,))

        model_bptt = LoopedMLP(
            input_dim,
            hidden_dim,
            output_dim,
            max_steps=max_steps,
            gradient_method="bptt",
            use_spectral_norm=False,
        )
        model_eq = LoopedMLP(
            input_dim,
            hidden_dim,
            output_dim,
            max_steps=max_steps,
            gradient_method="equilibrium",
            use_spectral_norm=False,
        )

        model_eq.load_state_dict(model_bptt.state_dict())

        criterion = nn.CrossEntropyLoss()

        model_bptt.zero_grad()
        loss_bptt = criterion(model_bptt(x), y)
        loss_bptt.backward()

        model_eq.zero_grad()
        loss_eq = criterion(model_eq(x), y)
        loss_eq.backward()

        print(f"Loss BPTT: {loss_bptt.item():.6f}, EqProp: {loss_eq.item():.6f}")
        self.assertAlmostEqual(loss_bptt.item(), loss_eq.item(), places=5)

        for (n1, p1), (n2, p2) in zip(
            model_bptt.named_parameters(), model_eq.named_parameters()
        ):
            if p1.grad is not None and p2.grad is not None:
                diff = (p1.grad - p2.grad).norm().item()
                scale = p1.grad.norm().item() + p2.grad.norm().item()
                if scale > 1e-9:
                    rel_err = diff / scale
                    # print(f"Param {n1}: rel_err={rel_err:.6f}")
                    self.assertLess(rel_err, 0.1, f"Gradient mismatch for {n1}")

    def test_conv_gradient_parity(self):
        print("\nTesting ConvEqProp Gradient Parity (BPTT vs Equilibrium)...")
        input_channels = 1
        hidden_channels = 8
        output_dim = 3
        batch_size = 2
        max_steps = 50  # Conv might need fewer steps to converge due to simpler dynamics in this small test

        torch.manual_seed(42)

        # 8x8 image
        x = torch.randn(batch_size, input_channels, 8, 8)
        y = torch.randint(0, output_dim, (batch_size,))

        # use_spectral_norm=False to simplify gradient flow for parity check
        model_bptt = ConvEqProp(
            input_channels,
            hidden_channels,
            output_dim,
            max_steps=max_steps,
            gradient_method="bptt",
            use_spectral_norm=False,
        )
        model_eq = ConvEqProp(
            input_channels,
            hidden_channels,
            output_dim,
            max_steps=max_steps,
            gradient_method="equilibrium",
            use_spectral_norm=False,
        )

        model_eq.load_state_dict(model_bptt.state_dict())

        criterion = nn.CrossEntropyLoss()

        model_bptt.zero_grad()
        loss_bptt = criterion(model_bptt(x), y)
        loss_bptt.backward()

        model_eq.zero_grad()
        loss_eq = criterion(model_eq(x), y)
        loss_eq.backward()

        print(f"Loss BPTT: {loss_bptt.item():.6f}, EqProp: {loss_eq.item():.6f}")
        self.assertAlmostEqual(loss_bptt.item(), loss_eq.item(), places=5)

        for (n1, p1), (n2, p2) in zip(
            model_bptt.named_parameters(), model_eq.named_parameters()
        ):
            if p1.grad is not None and p2.grad is not None:
                diff = (p1.grad - p2.grad).norm().item()
                scale = p1.grad.norm().item() + p2.grad.norm().item()
                if scale > 1e-9:
                    rel_err = diff / scale
                    # print(f"Param {n1}: rel_err={rel_err:.6f}")
                    # Allow slightly higher tolerance for Conv due to more complex graph/accumulation
                    self.assertLess(rel_err, 0.15, f"Gradient mismatch for {n1}")


if __name__ == "__main__":
    unittest.main()
