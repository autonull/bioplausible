"""
Leaderboard generator for the Bioplausible platform.

Tracks and ranks all model+propagator+optimizer combinations
across standardized benchmarks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LeaderboardEntry:
    """An entry in the benchmark leaderboard."""

    rank: int
    model: str
    propagator: Optional[str] = None
    optimizer: str = "adam"
    task: str = "mnist"
    accuracy: float = 0.0
    loss: float = 0.0
    bio_plausibility_score: float = 0.0
    requires_backward: bool = True
    params: int = 0
    energy_proxy: Optional[float] = None
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


class LeaderboardGenerator:
    """
    Generates and manages benchmark leaderboards.

    Ranks model+propagator+optimizer combinations by accuracy,
    with filters for bio-plausibility, energy efficiency, etc.
    """

    def __init__(self, output_dir: str = "leaderboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[LeaderboardEntry] = []

    def add_result(self, entry: LeaderboardEntry) -> None:
        """Add a result to the leaderboard."""
        self._entries.append(entry)

    def add_results(self, entries: List[LeaderboardEntry]) -> None:
        """Add multiple results."""
        self._entries.extend(entries)

    def get_leaderboard(
        self,
        task: Optional[str] = None,
        min_bio_score: float = 0.0,
        max_bio_score: float = 1.0,
        backward_only: Optional[bool] = None,
        top_k: int = 20,
    ) -> List[LeaderboardEntry]:
        """
        Get ranked leaderboard with filters.

        Args:
            task: Filter by task name.
            min_bio_score: Minimum bio-plausibility score.
            max_bio_score: Maximum bio-plausibility score.
            backward_only: If True, only include methods requiring backward pass.
            top_k: Number of top entries to return.

        Returns:
            Ranked list of leaderboard entries.
        """
        filtered = list(self._entries)

        if task:
            filtered = [e for e in filtered if e.task == task]
        if min_bio_score > 0:
            filtered = [
                e for e in filtered if e.bio_plausibility_score >= min_bio_score
            ]
        if max_bio_score < 1.0:
            filtered = [
                e for e in filtered if e.bio_plausibility_score <= max_bio_score
            ]
        if backward_only is not None:
            filtered = [e for e in filtered if e.requires_backward == backward_only]

        # Sort by accuracy descending
        filtered.sort(key=lambda e: e.accuracy, reverse=True)

        # Assign ranks
        for i, entry in enumerate(filtered[:top_k]):
            entry.rank = i + 1

        return filtered[:top_k]

    def save(self, path: Optional[str] = None) -> str:
        """Save leaderboard to JSON."""
        save_path = Path(path or self.output_dir / "leaderboard.json")
        data = [
            {
                "rank": e.rank,
                "model": e.model,
                "propagator": e.propagator,
                "optimizer": e.optimizer,
                "task": e.task,
                "accuracy": e.accuracy,
                "loss": e.loss,
                "bio_plausibility_score": e.bio_plausibility_score,
                "requires_backward": e.requires_backward,
                "params": e.params,
                "energy_proxy": e.energy_proxy,
                "timestamp": e.timestamp,
                "config": e.config,
            }
            for e in sorted(self._entries, key=lambda e: e.accuracy, reverse=True)
        ]
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Leaderboard saved: {save_path}")
        return str(save_path)

    def load(self, path: str) -> None:
        """Load leaderboard from JSON."""
        with open(path) as f:
            data = json.load(f)
        for item in data:
            self._entries.append(LeaderboardEntry(**item))
        logger.info(f"Loaded {len(data)} entries from {path}")

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._entries:
            return {"total": 0}
        return {
            "total": len(self._entries),
            "tasks": list(set(e.task for e in self._entries)),
            "models": list(set(e.model for e in self._entries)),
            "best_accuracy": max(e.accuracy for e in self._entries),
            "avg_accuracy": sum(e.accuracy for e in self._entries) / len(self._entries),
        }
