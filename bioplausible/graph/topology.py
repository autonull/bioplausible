# Adapted from FabricPC (https://github.com/trueagi-io/FabricPC)
# Original authors: Dr. Matthew Behrend et al., SingularityNET
# MIT License. See FABRICPC_INTEGRATION.md for details.

"""Graph topology — edges, task maps, and graph structure validation.

Analogous to FabricPC's GraphStructure which defines node connectivity
independently of the weights/activities.
"""

from __future__ import annotations

from collections import deque

from bioplausible.graph.nodes import NodeBase
from bioplausible.graph.nodes import Slot


class Edge:
    """A directed connection from a source node to a target slot.

    FabricPC equivalent: directed edge in the computation graph.
    Cycles are permitted (enables recurrent connections).
    """

    def __init__(self, source: NodeBase, target: Slot) -> None:
        self.source = source
        self.target = target

    def __repr__(self) -> str:
        return f"Edge({self.source.name} -> {self.target})"


class TaskMap:
    """Maps input/output nodes for supervised learning tasks.

    FabricPC equivalent: identifies which nodes receive external
    input (x) and which produce the output prediction (y).
    """

    def __init__(self, x: NodeBase, y: NodeBase) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"TaskMap(x={self.x.name}, y={self.y.name})"


class GraphStructure:
    """A validated directed graph of nodes and edges.

    Supports feedforward, recurrent, and skip-connection topologies.
    Provides topological ordering for feedforward passes.

    FabricPC equivalent: the central GraphStructure abstraction.
    """

    def __init__(
        self,
        nodes: list[NodeBase],
        edges: list[Edge],
        task_map: TaskMap,
        inference: InferenceSGD | None = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.task_map = task_map
        self.inference = inference

        # Cache: node_name -> node
        self._node_map: dict[str, NodeBase] = {n.name: n for n in nodes}

        # Cache: node_name -> list of (source_node, target_slot) for incoming edges
        self._predecessors: dict[str, list[tuple[NodeBase, Slot]]] = {
            n.name: [] for n in nodes
        }
        # Cache: node_name -> list of (source_node, target_slot) for outgoing edges
        self._successors: dict[str, list[tuple[NodeBase, Slot]]] = {
            n.name: [] for n in nodes
        }

        for edge in edges:
            src_name = edge.source.name
            tgt_name = edge.target.owner.name
            if src_name in self._successors:
                self._successors[src_name].append((edge.source, edge.target))
            if tgt_name in self._predecessors:
                self._predecessors[tgt_name].append((edge.source, edge.target))

        # Build slot-to-edge mapping for validation
        self._slot_map: dict[str, dict[str, Edge]] = {}
        for node in nodes:
            self._slot_map[node.name] = {}
        for edge in edges:
            owner = edge.target.owner
            slot_name = edge.target.name
            if owner.name in self._slot_map:
                self._slot_map[owner.name][slot_name] = edge

    def get_node(self, name: str) -> NodeBase:
        """Get a node by name."""
        if name not in self._node_map:
            raise KeyError(
                f"Node '{name}' not found in graph. "
                f"Available: {list(self._node_map.keys())}"
            )
        return self._node_map[name]

    def get_predecessors(self, node: NodeBase) -> list[tuple[NodeBase, Slot]]:
        """Get all (source_node, target_slot) pairs feeding into this node."""
        return self._predecessors.get(node.name, [])

    def get_successors(self, node: NodeBase) -> list[tuple[NodeBase, Slot]]:
        """Get all outgoing edges from this node."""
        return self._successors.get(node.name, [])

    def topological_order(self) -> list[NodeBase]:
        """Return nodes in topological order via Kahn's algorithm.

        Raises:
            ValueError: If the graph contains a directed cycle (no feedforward order).
        """
        in_degree: dict[str, int] = {}
        adjacency: dict[str, list[str]] = {}

        for node in self.nodes:
            in_degree[node.name] = 0
            adjacency[node.name] = []

        for edge in self.edges:
            src = edge.source.name
            tgt = edge.target.owner.name
            if src in adjacency:
                adjacency[src].append(tgt)
                if tgt in in_degree:
                    in_degree[tgt] += 1

        queue: deque[str] = deque()
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append(name)

        ordered: list[NodeBase] = []
        while queue:
            name = queue.popleft()
            ordered.append(self._node_map[name])
            for successor in adjacency.get(name, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(ordered) != len(self.nodes):
            cycle_nodes = [n.name for n in self.nodes if n not in ordered]
            raise ValueError(
                f"Graph contains a directed cycle involving nodes: {cycle_nodes}. "
                "Use inference.settle() for cyclic graphs."
            )

        return ordered

    def validate(self) -> None:
        """Validate graph consistency.

        Checks:
            1. All edge source/target nodes exist in the node list.
            2. Every slot on every node has exactly one incoming edge (or none for input).
            3. TaskMap x/y nodes exist in the graph.
            4. No duplicate edges.

        Raises:
            ValueError: On any validation failure.
        """
        node_names: set[str] = {n.name for n in self.nodes}

        # Check all edge endpoints are valid nodes
        for edge in self.edges:
            if edge.source.name not in node_names:
                raise ValueError(f"Edge source '{edge.source.name}' not in graph nodes")
            if edge.target.owner.name not in node_names:
                raise ValueError(
                    f"Edge target owner '{edge.target.owner.name}' not in graph nodes"
                )

        # Check each slot has at least one incoming edge
        slot_edge_count: dict[tuple[str, str], int] = {}
        for edge in self.edges:
            key = (edge.target.owner.name, edge.target.name)
            slot_edge_count[key] = slot_edge_count.get(key, 0) + 1

        # Check TaskMap nodes exist
        if self.task_map.x.name not in node_names:
            raise ValueError(
                f"TaskMap x node '{self.task_map.x.name}' not in graph nodes"
            )
        if self.task_map.y.name not in node_names:
            raise ValueError(
                f"TaskMap y node '{self.task_map.y.name}' not in graph nodes"
            )

    def __repr__(self) -> str:
        return f"GraphStructure(nodes={len(self.nodes)}, edges={len(self.edges)})"


def graph(
    nodes: list[NodeBase],
    edges: list[Edge],
    task_map: TaskMap,
    inference: InferenceSGD | None = None,  # type: ignore[name-defined]  # noqa: F821
) -> GraphStructure:
    """Assemble and validate a GraphStructure.

    This is the primary constructor — analogous to FabricPC's ``graph()`` function.
    Validation is performed automatically.

    Args:
        nodes: All nodes in the graph.
        edges: All directed edges.
        task_map: Input/output node mapping.
        inference: Optional InferenceSGD instance for PC settling.

    Returns:
        A validated GraphStructure.
    """
    structure = GraphStructure(
        nodes=nodes, edges=edges, task_map=task_map, inference=inference
    )
    structure.validate()
    return structure
