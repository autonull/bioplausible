"""Persistent node database for evolution experiments."""

import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from ..utils.structures import Node
from .algorithms import BaseSampler, get_sampler
from .faiss_index import FAISSIndex
from .embedding import EmbeddingService


class Database:
    """
    Persistent experiment database.

    Features:
    - CRUD operations for nodes
    - Configurable sampling strategies
    - Embedding-based similarity search
    - Local persistence to disk
    """

    def __init__(
        self,
        storage_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        sampling_algorithm: str = "ucb1",
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        faiss_index_type: str = "IP",
        max_size: Optional[int] = None,
    ):
        """
        Args:
            storage_dir: Storage directory.
            embedding_model: Embedding model name.
            embedding_dim: Embedding dimension.
            sampling_algorithm: Sampling algorithm name.
            sampling_kwargs: Sampling algorithm parameters.
            faiss_index_type: FAISS index type.
            max_size: Optional maximum number of nodes to keep.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.lock = RLock()

        self.nodes: Dict[int, Node] = {}
        self.next_id = 0
        self.max_size = max_size

        self.embedding = EmbeddingService(model_name=embedding_model)

        self.faiss = FAISSIndex(
            dimension=embedding_dim,
            index_type=faiss_index_type,
            storage_path=self.storage_dir / "faiss",
        )

        self.sampling_algorithm = sampling_algorithm
        self.sampling_kwargs = sampling_kwargs or {}
        self.default_sampler = get_sampler(sampling_algorithm, **self.sampling_kwargs)

        self._load()

    def sample(
        self,
        n: int,
        algorithm: Optional[str] = None,
        **kwargs,
    ) -> List[Node]:
        """
        Sample nodes from the database.

        Args:
            n: Number of nodes to sample.
            algorithm: Optional override for the sampling algorithm.
            **kwargs: Extra parameters for the sampler.

        Returns:
            A list of sampled nodes.
        """
        with self.lock:
            nodes = list(self.nodes.values())

            if algorithm:
                sampler = get_sampler(algorithm, **kwargs)
            else:
                sampler = self.default_sampler

            selected = sampler.sample(nodes, n)
            self._save()

            return selected

    def add(self, node: Node) -> int:
        """
        Add a node to the database.

        If `max_size` is set and the database is full, the lowest-scoring node
        is removed first. Ties are broken by the smallest id.
        """
        with self.lock:
            if self.max_size is not None and len(self.nodes) >= self.max_size:
                self._remove_worst_node()

            node.id = self.next_id
            self.next_id += 1

            self.nodes[node.id] = node
            self.default_sampler.on_node_added(node)

            text = node.get_context_text()
            if text:
                vector = self.embedding.encode(text)
                self.faiss.add(node.id, vector)

            self._save()
            return node.id

    def add_batch(self, nodes: List[Node]) -> List[int]:
        return [self.add(node) for node in nodes]

    def remove(self, node_id: int) -> bool:
        with self.lock:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]
            self.default_sampler.on_node_removed(node)

            del self.nodes[node_id]
            self.faiss.remove(node_id)
            self._save()
            return True

    def remove_batch(self, node_ids: List[int]) -> int:
        return sum(1 for nid in node_ids if self.remove(nid))

    def _remove_worst_node(self):
        """Remove the lowest-scoring node, breaking ties by the smallest id."""
        if not self.nodes:
            return

        worst_node_id = min(
            self.nodes.keys(),
            key=lambda node_id: (self.nodes[node_id].score, node_id)
        )

        self.remove(worst_node_id)

    def get(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_all(self) -> List[Node]:
        return list(self.nodes.values())

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Node]:
        query_vector = self.embedding.encode(query)
        results = self.faiss.search(query_vector, top_k, score_threshold)

        nodes = []
        for node_id, score in results:
            node = self.nodes.get(node_id)
            if node:
                nodes.append(node)

        return nodes

    def reset(self):
        """Clear the database and sampler state."""
        with self.lock:
            if hasattr(self.default_sampler, "reset"):
                self.default_sampler.reset()

            self.nodes.clear()
            self.next_id = 0
            self.faiss.reset()

            data_file = self.storage_dir / "nodes.json"
            if data_file.exists():
                data_file.unlink()

    def get_sampler_stats(self) -> Optional[Dict[str, Any]]:
        """
        Return sampler statistics if the sampler exposes them.

        Different samplers may report different data, for example island
        populations or sampler-specific counters.
        """
        if hasattr(self.default_sampler, "get_island_stats"):
            with self.lock:
                nodes = list(self.nodes.values())
                return self.default_sampler.get_island_stats(nodes)

        if hasattr(self.default_sampler, "get_stats"):
            with self.lock:
                nodes = list(self.nodes.values())
                return self.default_sampler.get_stats(nodes)

        return None

    def call_sampler_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a custom sampler method if it exists.

        This allows access to sampler-specific functionality such as
        `sample_from_island(...)`.
        """
        if not hasattr(self.default_sampler, method_name):
            raise AttributeError(
                f"Sampler '{self.sampling_algorithm}' does not have method '{method_name}'"
            )

        method = getattr(self.default_sampler, method_name)

        with self.lock:
            import inspect
            sig = inspect.signature(method)
            if "nodes" in sig.parameters:
                kwargs["nodes"] = list(self.nodes.values())

            return method(*args, **kwargs)

    def _save(self):
        """Persist database state to disk."""
        data_file = self.storage_dir / "nodes.json"

        data = {
            "next_id": self.next_id,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
        }

        if hasattr(self.default_sampler, "get_state"):
            data["sampler_state"] = self.default_sampler.get_state()

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.faiss.save()

    def _load(self):
        """Load database state from disk if available."""
        data_file = self.storage_dir / "nodes.json"

        if not data_file.exists():
            return

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.next_id = data.get("next_id", 0)

        for node_id, node_data in data.get("nodes", {}).items():
            node = Node.from_dict(node_data)
            self.nodes[int(node_id)] = node

        if hasattr(self.default_sampler, "load_state") and "sampler_state" in data:
            self.default_sampler.load_state(data["sampler_state"])

        if hasattr(self.default_sampler, "rebuild_from_nodes"):
            self.default_sampler.rebuild_from_nodes(list(self.nodes.values()))

    @property
    def size(self) -> int:
        return len(self.nodes)

    def __len__(self) -> int:
        return self.size
