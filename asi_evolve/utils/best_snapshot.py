"""Utilities for persisting the best-scoring step outputs."""

import json
from pathlib import Path
from threading import Lock
from typing import List, Optional

from .structures import Node


class BestSnapshotManager:
    """Maintain a `steps/best` directory with the strongest snapshots."""

    def __init__(self, steps_dir: Path, logger=None):
        self.steps_dir = Path(steps_dir)
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.steps_dir / "best"
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.best_score = float("-inf")
        self._lock = Lock()

    def init_from_nodes(self, nodes: List[Node]) -> None:
        """Initialize the best-score tracker from existing nodes."""
        if not nodes:
            return

        best_node = max(nodes, key=lambda n: n.score)
        with self._lock:
            self.best_score = best_node.score

    def update_if_better(
        self,
        node: Node,
        step_name: str,
        source_step_dir: Optional[Path] = None,
    ) -> bool:
        """Write a new snapshot if the provided node improves on the best score."""
        with self._lock:
            if node.score <= self.best_score:
                return False

            self.best_score = node.score
            self._write_snapshot(node, step_name=step_name, source_step_dir=source_step_dir)
            if self.logger:
                self.logger.info(
                    f"Updated best snapshot: {node.name} (score={node.score:.4f}) -> {self.best_dir / step_name}"
                )
            return True

    def _write_snapshot(self, node: Node, step_name: str, source_step_dir: Optional[Path]) -> None:
        """Persist code and results for a best-performing step."""
        best_step_dir = self.best_dir / step_name
        best_step_dir.mkdir(parents=True, exist_ok=True)

        code_file = best_step_dir / "code"
        code_file.write_text(node.code or "", encoding="utf-8")

        best_results_file = best_step_dir / "results.json"
        source_results_file = source_step_dir / "results.json" if source_step_dir else None
        if source_results_file and source_results_file.exists():
            best_results_file.write_text(source_results_file.read_text(encoding="utf-8"), encoding="utf-8")
            return

        with open(best_results_file, "w", encoding="utf-8") as f:
            json.dump(node.results or {}, f, ensure_ascii=False, indent=2)
