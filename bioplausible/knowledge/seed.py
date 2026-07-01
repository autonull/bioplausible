import json
import os
from typing import Any, Dict, List

# Statically seeded knowledge base containing key findings and empirical rules
KNOWLEDGE_BASE_SEED = [
    {
        "id": "KB-001",
        "topic": "Scaling",
        "model_family": "eqprop",
        "finding": "O(1) memory scaling for BPTT equivalent",
        "details": (
            "Equilibrium Propagation requires constant memory "
            "regardless of trajectory length, unlike BPTT which scales O(T)."
        ),
        "confidence": 0.95,
        "tags": ["memory", "scaling", "eqprop"],
    },
    {
        "id": "KB-002",
        "topic": "Architecture",
        "model_family": "tile_eq",
        "finding": "Optimal 2D grid layout improves locality",
        "details": (
            "TileEQ variants arranged in a 2D locally connected grid "
            "demonstrate superior scaling on neuromorphic simulators "
            "vs fully connected counterparts."
        ),
        "confidence": 0.85,
        "tags": ["architecture", "neuromorphic", "local-learning"],
    },
    {
        "id": "KB-003",
        "topic": "Optimization",
        "model_family": "forward_forward",
        "finding": "Layer-local goodness thresholds",
        "details": (
            "A threshold of 2.0 provides stable contrastive separation "
            "on MNIST-level tasks without causing early layer saturation."
        ),
        "confidence": 0.80,
        "tags": ["hyperparams", "forward-forward", "thresholds"],
    },
]


class KnowledgeBase:
    """
    Structured repository of findings across experiments.
    """

    def __init__(
        self, storage_path: str = "knowledgebase.json", load_seed: bool = False
    ):
        self.storage_path = storage_path
        self.load_seed = load_seed
        self.findings = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.findings = json.load(f)
        else:
            if self.load_seed:
                self.findings = list(KNOWLEDGE_BASE_SEED)
            else:
                self.findings = []
            self._save()

    def _save(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.findings, f, indent=4)

    def add_finding(
        self,
        topic: str,
        model_family: str,
        finding: str,
        details: str,
        confidence: float,
        tags: List[str],
    ):
        new_id = f"KB-{len(self.findings) + 1:03d}"
        entry = {
            "id": new_id,
            "topic": topic,
            "model_family": model_family,
            "finding": finding,
            "details": details,
            "confidence": confidence,
            "tags": tags,
        }
        self.findings.append(entry)
        self._save()
        return new_id

    def query(self, tag: str = None, model_family: str = None) -> List[Dict[str, Any]]:
        results = self.findings
        if tag:
            results = [r for r in results if tag in r.get("tags", [])]
        if model_family:
            results = [r for r in results if r.get("model_family") == model_family]
        return results


# Singleton instance
DEFAULT_KB = KnowledgeBase()
