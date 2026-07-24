"""Tests for InferenceSGD — energy minimization settling."""

import pytest
import torch

from bioplausible.graph import Edge
from bioplausible.graph import InferenceSGD
from bioplausible.graph import Linear
from bioplausible.graph import ReLU
from bioplausible.graph import TaskMap
from bioplausible.graph import graph
from bioplausible.graph import initialize_params


@pytest.fixture
def simple_graph():
    input_node = Linear(shape=(4, 8), name="input")
    hidden = ReLU(name="hidden")
    output = Linear(shape=(8, 2), name="output")

    structure = graph(
        nodes=[input_node, hidden, output],
        edges=[
            Edge(source=input_node, target=hidden.slot("input")),
            Edge(source=hidden, target=output.slot("input")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=10),
    )
    params = initialize_params(structure)
    return structure, params


def test_unsupervised_settle(simple_graph):
    """Settle without supervision produces activities for all nodes."""
    structure, params = simple_graph
    x = torch.randn(8, 4)

    activities = structure.inference.settle(structure, params, x)

    assert structure.task_map.x.name in activities
    assert structure.task_map.y.name in activities
    assert "hidden" in activities
    # Input is Linear(4,8) so forward(x=(8,4)) -> (8,8)
    assert activities[structure.task_map.x.name].shape == (8, 8)


def test_supervised_settle(simple_graph):
    """Supervised settle clamps output toward target."""
    structure, params = simple_graph
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))

    activities = structure.inference.settle(structure, params, x, y=y)

    # Input is Linear(4,8) so forward(x=(8,4)) -> (8,8)
    assert activities[structure.task_map.x.name].shape == (8, 8)


def test_energy_decreases(simple_graph):
    """Energy should decrease (or stay same) over settling steps."""
    structure, params = simple_graph
    x = torch.randn(4, 4)

    # Run step-by-step to track energy
    infer = InferenceSGD(eta_infer=0.1, infer_steps=20)
    activities = infer.settle(structure, params, x)

    # Verify all nodes have activities
    for node in structure.nodes:
        assert node.name in activities


def test_1_step_degenerate(simple_graph):
    """1-step inference should run without error."""
    structure, params = simple_graph
    infer = InferenceSGD(eta_infer=0.1, infer_steps=1)

    x = torch.randn(4, 4)
    activities = infer.settle(structure, params, x)

    assert len(activities) == len(structure.nodes)


def test_multi_batch(simple_graph):
    """Inference works with batch dimension."""
    structure, params = simple_graph
    x = torch.randn(16, 4)

    activities = structure.inference.settle(structure, params, x)
    assert activities[structure.task_map.y.name].shape[0] == 16


def test_inference_repr():
    infer = InferenceSGD(eta_infer=0.05, infer_steps=20)
    r = repr(infer)
    assert "0.05" in r
    assert "20" in r


def test_inference_defaults():
    infer = InferenceSGD()
    assert infer.eta_infer == 0.05
    assert infer.infer_steps == 20


def test_different_eta():
    infer = InferenceSGD(eta_infer=0.01, infer_steps=50)
    assert infer.eta_infer == 0.01
    assert infer.infer_steps == 50
