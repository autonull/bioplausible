"""Greedy sampling strategy."""

from typing import List, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class GreedySampler(BaseSampler):
    """Return the highest-scoring nodes first."""

    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        if not nodes:
            return []
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return sorted_nodes[:n]
