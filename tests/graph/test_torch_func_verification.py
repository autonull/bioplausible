"""Critical verification: torch.func.grad on Linear node.

This is the single most important test — if torch.func.grad does not work
on our node forward functions, the entire PC training approach fails.
"""

import torch

from bioplausible.graph.nodes import Linear


def test_torch_func_grad_on_linear_node():
    node = Linear(shape=(784, 256), name="test")
    params = node.initialize_params(torch.Generator().manual_seed(0))
    x = torch.randn(32, 784)

    def energy(p):
        out = node.forward(input=x, weight=p["weight"], bias=p["bias"])
        return (out**2).sum()

    grads = torch.func.grad(energy)(params)
    assert grads["weight"].shape == params["weight"].shape
    assert grads["bias"].shape == params["bias"].shape
