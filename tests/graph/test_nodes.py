"""Tests for graph node implementations."""

import pytest
import torch

from bioplausible.graph.nodes import Linear, ReLU, Slot, Tanh


class TestSlot:
    def test_slot_creation(self):
        node = Linear(shape=(10, 5), name="test")
        s = Slot("input", node)
        assert s.name == "input"
        assert s.owner == node
        assert repr(s) == "Slot(test.input)"

    def test_slot_on_node(self):
        node = Linear(shape=(10, 5), name="test")
        s = node.slot("input")
        assert s.name == "input"
        assert s.owner == node


class TestLinear:
    def test_creation(self):
        node = Linear(shape=(784, 256), name="fc1")
        assert node.name == "fc1"
        assert node.shape == (784, 256)

    def test_slots(self):
        node = Linear(shape=(784, 256), name="fc1")
        slots = node.get_slots()
        assert "input" in slots
        assert len(slots) == 1

    def test_slot_method(self):
        node = Linear(shape=(784, 256), name="fc1")
        s = node.slot("input")
        assert isinstance(s, Slot)

    def test_slot_missing(self):
        node = Linear(shape=(784, 256), name="fc1")
        with pytest.raises(KeyError):
            node.slot("nonexistent")

    def test_forward_shape(self):
        node = Linear(shape=(784, 256), name="fc1")
        params = node.initialize_params(torch.Generator().manual_seed(0))
        x = torch.randn(32, 784)
        out = node.forward(
            **{"input": x, "weight": params["weight"], "bias": params["bias"]}
        )
        assert out.shape == (32, 256)

    def test_forward_no_weight(self):
        node = Linear(shape=(784, 256), name="fc1")
        x = torch.randn(32, 784)
        with pytest.raises(TypeError):
            node.forward(**{"input": x})

    def test_initialize_params_shapes(self):
        node = Linear(shape=(784, 256), name="fc1")
        params = node.initialize_params(torch.Generator().manual_seed(0))
        assert "weight" in params
        assert "bias" in params
        assert params["weight"].shape == (256, 784)
        assert params["bias"].shape == (256,)

    def test_initialize_params_deterministic(self):
        node = Linear(shape=(784, 256), name="fc1")
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        p1 = node.initialize_params(gen1)
        p2 = node.initialize_params(gen2)
        assert torch.equal(p1["weight"], p2["weight"])
        assert torch.equal(p1["bias"], p2["bias"])

    def test_torch_func_grad_on_params(self):
        node = Linear(shape=(784, 256), name="test")
        params = node.initialize_params(torch.Generator().manual_seed(0))
        x = torch.randn(32, 784)

        def energy(p):
            out = node.forward(**{"input": x, "weight": p["weight"], "bias": p["bias"]})
            return (out**2).sum()

        grads = torch.func.grad(energy)(params)
        assert grads["weight"].shape == params["weight"].shape
        assert grads["bias"].shape == params["bias"].shape

    def test_repr(self):
        node = Linear(shape=(784, 256), name="fc1")
        assert "Linear" in repr(node)
        assert "fc1" in repr(node)


class TestReLU:
    def test_creation(self):
        node = ReLU(name="relu1")
        assert node.name == "relu1"

    def test_forward(self):
        node = ReLU(name="relu1")
        x = torch.randn(32, 256)
        out = node.forward(**{"input": x})
        assert out.shape == (32, 256)
        assert (out >= 0).all()

    def test_no_params(self):
        node = ReLU(name="relu1")
        params = node.initialize_params(torch.Generator().manual_seed(0))
        assert params == {}

    def test_slots(self):
        node = ReLU(name="relu1")
        slots = node.get_slots()
        assert "input" in slots


class TestTanh:
    def test_creation(self):
        node = Tanh(name="tanh1")
        assert node.name == "tanh1"

    def test_forward(self):
        node = Tanh(name="tanh1")
        x = torch.randn(32, 256)
        out = node.forward(**{"input": x})
        assert out.shape == (32, 256)
        assert (out >= -1).all()
        assert (out <= 1).all()

    def test_no_params(self):
        node = Tanh(name="tanh1")
        params = node.initialize_params(torch.Generator().manual_seed(0))
        assert params == {}
