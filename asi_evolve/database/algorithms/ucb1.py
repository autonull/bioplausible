"""UCB1-based sampling strategy."""

import math
import random
from typing import List, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class UCB1Sampler(BaseSampler):
    """
    Balance exploration and exploitation with UCB1.

    UCB1 = normalized_score + c * sqrt(ln(N) / n_i)
    """

    def __init__(self, c: float = 1.414):
        """
        Args:
            c: Exploration coefficient.
        """
        self.c = c

    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        if not nodes:
            return []

        n = min(n, len(nodes))

        total_visits = sum(node.visit_count for node in nodes)
        if total_visits == 0:
            return random.sample(nodes, n)

        scores = [node.score for node in nodes if node.visit_count > 0]
        if not scores:
            return random.sample(nodes, n)

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        ucb1_values = []
        for node in nodes:
            if node.visit_count == 0:
                ucb1 = float("inf")
            else:
                normalized_score = (node.score - min_score) / score_range
                exploration = self.c * math.sqrt(math.log(total_visits) / node.visit_count)
                ucb1 = normalized_score + exploration

            ucb1_values.append((node, ucb1))

        ucb1_values.sort(key=lambda x: x[1], reverse=True)

        selected = [node for node, _ in ucb1_values[:n]]
        for node in selected:
            node.visit_count += 1

        return selected
