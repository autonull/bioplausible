"""
Multi-Objective Metrics

Implements Pareto dominance, non-dominated sorting, and composite scoring.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class TrialMetrics:
    """Metrics for a single trial."""

    trial_id: int
    model_name: str
    config: Dict[str, Any]

    # Objectives (4D)
    accuracy: float  # Maximize (0-1)
    perplexity: float  # Minimize
    iteration_time: float  # Minimize (seconds)
    param_count: float  # Minimize (millions)

    # Training metadata
    epochs_completed: int
    final_loss: float
    status: str  # 'completed', 'failed', 'running'

    def __post_init__(self):
        # Normalize for comparison
        self.objectives = np.array(
            [
                self.accuracy,  # Higher is better
                -self.perplexity,  # Convert to maximization (higher is better)
                -self.iteration_time,  # Convert to maximization
                -self.param_count,  # Convert to maximization
            ]
        )

    def dominates(self, other: "TrialMetrics") -> bool:
        """Check if this trial Pareto-dominates another.

        A dominates B if A is >= B on all objectives AND strictly > on at least one.
        """
        better_or_equal = np.all(self.objectives >= other.objectives)
        strictly_better = np.any(self.objectives > other.objectives)
        return better_or_equal and strictly_better

    def composite_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted composite score for ranking.

        Default weights balance all objectives equally.
        """
        if weights is None:
            weights = {"accuracy": 0.4, "perplexity": 0.3, "speed": 0.2, "params": 0.1}

        # Normalize each objective to [0, 1] scale (roughly)
        norm_acc = self.accuracy  # Already 0-1
        norm_ppl = max(0, 1 - self.perplexity / 10.0)  # PPL ~0-10
        norm_speed = max(0, 1 - self.iteration_time / 1.0)  # Time ~0-1s
        norm_params = max(0, 1 - self.param_count / 10.0)  # Params ~0-10M

        score = (
            weights["accuracy"] * norm_acc
            + weights["perplexity"] * norm_ppl
            + weights["speed"] * norm_speed
            + weights["params"] * norm_params
        )
        return score


def non_dominated_sort(trials: List[TrialMetrics]) -> List[List[int]]:
    """Non-dominated sorting (NSGA-II).

    Returns:
        fronts: List of fronts, where each front is a list of trial indices.
                Front 0 = Pareto frontier (best).
    """
    n = len(trials)
    domination_count = np.zeros(n, dtype=int)  # How many dominate this trial
    dominated_by = [[] for _ in range(n)]  # Which trials does this dominate

    # Build domination relationships
    for i in range(n):
        for j in range(i + 1, n):
            if trials[i].dominates(trials[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif trials[j].dominates(trials[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

    # Extract fronts
    fronts = []
    current_front = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []

        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)

        current_front = next_front

    return fronts


def crowding_distance(
    trials: List[TrialMetrics], front_indices: List[int]
) -> np.ndarray:
    """Calculate crowding distance for diversity preservation.

    Higher distance = more isolated = should be preserved.
    """
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    n_objectives = 4

    for obj_idx in range(n_objectives):
        # Sort by this objective
        sorted_indices = sorted(
            range(n), key=lambda i: trials[front_indices[i]].objectives[obj_idx]
        )

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # Calculate distances for interior points
        obj_range = (
            trials[front_indices[sorted_indices[-1]]].objectives[obj_idx]
            - trials[front_indices[sorted_indices[0]]].objectives[obj_idx]
        )

        if obj_range > 0:
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    trials[front_indices[sorted_indices[i + 1]]].objectives[obj_idx]
                    - trials[front_indices[sorted_indices[i - 1]]].objectives[obj_idx]
                ) / obj_range

    return distances


def get_pareto_frontier(trials: List[TrialMetrics]) -> List[int]:
    """Get indices of trials on the Pareto frontier (front 0)."""
    fronts = non_dominated_sort(trials)
    return fronts[0] if fronts else []


def rank_trials(
    trials: List[TrialMetrics], top_k: int = None
) -> List[Tuple[int, float]]:
    """Rank trials by composite score.

    Returns:
        List of (trial_index, score) tuples, sorted best to worst.
    """
    scores = [(i, trial.composite_score()) for i, trial in enumerate(trials)]
    scores.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        scores = scores[:top_k]

    return scores
