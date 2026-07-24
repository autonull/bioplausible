"""Tests for graph topology — edges, GraphStructure, validation."""

import pytest

from bioplausible.graph.nodes import Linear
from bioplausible.graph.nodes import ReLU
from bioplausible.graph.topology import Edge
from bioplausible.graph.topology import GraphStructure
from bioplausible.graph.topology import TaskMap
from bioplausible.graph.topology import graph


@pytest.fixture
def simple_nodes():
    input_node = Linear(shape=(10, 20), name="input")
    hidden = ReLU(name="hidden")
    output = Linear(shape=(20, 5), name="output")
    return input_node, hidden, output


class TestEdge:
    def test_edge_creation(self):
        a = Linear(shape=(10, 20), name="a")
        b = Linear(shape=(20, 5), name="b")
        e = Edge(source=a, target=b.slot("input"))
        assert e.source == a
        assert e.target == b.slot("input")

    def test_edge_repr(self):
        a = Linear(shape=(10, 20), name="a")
        b = Linear(shape=(20, 5), name="b")
        e = Edge(source=a, target=b.slot("input"))
        assert "a" in repr(e)
        assert "b" in repr(e)


class TestTaskMap:
    def test_task_map_creation(self):
        a = Linear(shape=(10, 20), name="x")
        b = Linear(shape=(20, 5), name="y")
        tm = TaskMap(x=a, y=b)
        assert tm.x == a
        assert tm.y == b

    def test_task_map_repr(self):
        a = Linear(shape=(10, 20), name="input")
        b = Linear(shape=(20, 5), name="output")
        tm = TaskMap(x=a, y=b)
        assert "input" in repr(tm)
        assert "output" in repr(tm)


class TestGraphStructure:
    def test_feedforward_graph(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        assert len(structure.nodes) == 3
        assert len(structure.edges) == 2

    def test_topological_order(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        order = structure.topological_order()
        names = [n.name for n in order]
        assert names.index("input") < names.index("hidden")
        assert names.index("hidden") < names.index("output")

    def test_skip_connection(self):
        a = Linear(shape=(10, 20), name="a")
        b = ReLU(name="b")
        c = Linear(shape=(20, 5), name="c")
        structure = graph(
            nodes=[a, b, c],
            edges=[
                Edge(source=a, target=b.slot("input")),
                Edge(source=b, target=c.slot("input")),
                Edge(source=a, target=c.slot("input")),  # Skip connection
            ],
            task_map=TaskMap(x=a, y=c),
        )
        order = structure.topological_order()
        names = [n.name for n in order]
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    def test_cyclic_graph_topological_fails(self):
        a = Linear(shape=(10, 20), name="a")
        b = Linear(shape=(20, 10), name="b")
        structure = graph(
            nodes=[a, b],
            edges=[
                Edge(source=a, target=b.slot("input")),
                Edge(source=b, target=a.slot("input")),
            ],
            task_map=TaskMap(x=a, y=b),
        )
        with pytest.raises(ValueError, match="directed cycle"):
            structure.topological_order()

    def test_self_recurrent(self):
        a = Linear(shape=(10, 10), name="a")
        # Self-loop is permitted in the graph (cycles allowed for PC)
        # But topological_order should fail
        structure = GraphStructure(
            nodes=[a],
            edges=[Edge(source=a, target=a.slot("input"))],
            task_map=TaskMap(x=a, y=a),
        )
        structure.validate()
        with pytest.raises(ValueError, match="directed cycle"):
            structure.topological_order()

    def test_get_predecessors(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        preds = structure.get_predecessors(output)
        assert len(preds) == 1
        assert preds[0][0] == hidden

    def test_get_node(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        assert structure.get_node("input") == input_node

    def test_get_node_missing(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        with pytest.raises(KeyError):
            structure.get_node("nonexistent")


class TestValidation:
    def test_validate_valid(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        structure.validate()  # Should not raise

    def test_duplicate_slot_edge_allowed(self, simple_nodes):
        """Multiple edges to the same slot are allowed (inputs are summed)."""
        input_node, hidden, output = simple_nodes
        g = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=output, target=hidden.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        assert len(g.edges) == 2
        # Hidden now has 2 predecessors
        assert len(g.get_predecessors(hidden)) == 2

    def test_validate_missing_source_node(self):
        a = Linear(shape=(10, 20), name="a")
        b = Linear(shape=(20, 5), name="b")
        orphan = Linear(shape=(5, 5), name="orphan")
        with pytest.raises(ValueError, match="not in graph"):
            graph(
                nodes=[a, b],
                edges=[Edge(source=orphan, target=b.slot("input"))],
                task_map=TaskMap(x=a, y=b),
            )

    def test_validate_missing_target_owner(self):
        a = Linear(shape=(10, 20), name="a")
        b = Linear(shape=(20, 5), name="b")
        orphan = Linear(shape=(5, 5), name="orphan")
        with pytest.raises(ValueError, match="not in graph"):
            graph(
                nodes=[a, b],
                edges=[Edge(source=a, target=orphan.slot("input"))],
                task_map=TaskMap(x=a, y=b),
            )

    def test_validate_missing_taskmap_node(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        fake = Linear(shape=(5, 5), name="fake")
        with pytest.raises(ValueError, match="not in graph"):
            graph(
                nodes=[input_node, hidden, output],
                edges=[
                    Edge(source=input_node, target=hidden.slot("input")),
                    Edge(source=hidden, target=output.slot("input")),
                ],
                task_map=TaskMap(x=input_node, y=fake),
            )


class TestGraphHelper:
    def test_graph_factory(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        g = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        assert isinstance(g, GraphStructure)
        assert len(g.nodes) == 3

    def test_graph_repr(self, simple_nodes):
        input_node, hidden, output = simple_nodes
        g = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("input")),
                Edge(source=hidden, target=output.slot("input")),
            ],
            task_map=TaskMap(x=input_node, y=output),
        )
        r = repr(g)
        assert "GraphStructure" in r
        assert "3" in r
        assert "2" in r
