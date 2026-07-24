"""
Target Propagation Models
==========================

Difference Target Propagation model for the model zoo.
"""

import math
from typing import Dict

import torch
import torch.nn as nn

from bioplausible.core.registry import register_model


class DTPLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(in_features, out_features), nn.Tanh()
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(out_features, in_features), nn.Tanh()
        )
        self.opt_f = torch.optim.Adam(self.forward_net.parameters(), lr=0.001)
        self.opt_g = torch.optim.Adam(self.inverse_net.parameters(), lr=0.001)


@register_model("diff_target_prop")
class DifferenceTargetProp(nn.Module):
    """
    Difference Target Propagation (Lee et al. 2015).
    Propagates targets (not gradients) backward using learned approximate inverses.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ):
        super().__init__()
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList([DTPLayer(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(DTPLayer(hidden_dim, hidden_dim))
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        self.out_opt = torch.optim.Adam(self.out_layer.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers=2,
        device="cpu",
        task_type="vision",
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        ).to(device)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward_net(h)
        return self.out_layer(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        hs = [x]
        h = x
        for layer in self.layers:
            h = layer.forward_net(h)
            hs.append(h)
        out = self.out_layer(h)

        loss = self.criterion(out, y)
        self.out_opt.zero_grad()
        loss.backward()
        self.out_opt.step()

        t = h.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out_t = self.out_layer(t)
            loss_t = self.criterion(out_t, y)
            grad_t = torch.autograd.grad(loss_t, t)[0]

        with torch.no_grad():
            t_target = h - 0.1 * grad_t

        targets = [t_target]

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i > 0:
                h_prev = hs[i]
                h_curr = hs[i + 1]
                t_curr = targets[-1]

                with torch.no_grad():
                    t_prev = (
                        h_prev - layer.inverse_net(h_curr) + layer.inverse_net(t_curr)
                    )
                    targets.append(t_prev)

            t_curr = targets[-len(targets)]
            h_prev = hs[i].detach()
            layer.opt_f.zero_grad()
            pred_h = layer.forward_net(h_prev)
            loss_f = nn.functional.mse_loss(pred_h, t_curr)
            loss_f.backward()
            layer.opt_f.step()

            h_curr = hs[i + 1].detach()
            layer.opt_g.zero_grad()
            noise = torch.randn_like(h_curr) * 0.1
            pred_noise = layer.inverse_net(h_curr + noise)
            loss_g = nn.functional.mse_loss(
                pred_noise, h_prev + torch.randn_like(h_prev) * 0.1
            )
            loss_g.backward()
            layer.opt_g.step()

        acc = (out.argmax(1) == y).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}
