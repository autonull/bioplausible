"""Base interface for node samplers."""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.structures import Node


class BaseSampler(ABC):
    """Abstract sampler used by the experiment database."""

    @abstractmethod
    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        """
        Sample a subset of nodes.

        Args:
            nodes: All candidate nodes.
            n: Number of nodes to return.

        Returns:
            A list of sampled nodes.
        """
        pass

    def on_node_added(self, node: "Node") -> None:
        """
        Hook called when a node is added.

        Args:
            node: The newly added node.
        """
        pass

    def on_node_removed(self, node: "Node") -> None:
        """
        Hook called when a node is removed.

        Args:
            node: The removed node.
        """
        pass
