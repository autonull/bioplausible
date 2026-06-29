"""Random sampling strategy."""

import random
from typing import List, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class RandomSampler(BaseSampler):
    """Sample nodes uniformly at random."""

    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        if not nodes:
            return []
        n = min(n, len(nodes))
        return random.sample(nodes, n)
