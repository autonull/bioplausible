"""FAISS index wrapper."""

import pickle
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSIndex:
    """Manage FAISS-based vector indexing for similarity search."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "IP",
        storage_path: Optional[Path] = None,
    ):
        """
        Args:
            dimension: Vector dimension.
            index_type: Index type, either `IP` or `L2`.
            storage_path: Optional persistence path.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None
        self.lock = RLock()

        if index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        self.next_idx = 0

        if self.storage_path:
            self._load()

    def add(self, node_id: int, vector: np.ndarray):
        """
        Add a vector to the index.

        Args:
            node_id: Node identifier.
            vector: Vector to index.
        """
        with self.lock:
            if node_id in self.id_to_idx:
                return

            vector = self._normalize(vector.reshape(1, -1).astype(np.float32))
            self.index.add(vector)

            self.id_to_idx[node_id] = self.next_idx
            self.idx_to_id[self.next_idx] = node_id
            self.next_idx += 1

    def add_batch(self, node_ids: List[int], vectors: np.ndarray):
        """Add a batch of vectors."""
        with self.lock:
            for i, node_id in enumerate(node_ids):
                if node_id not in self.id_to_idx:
                    self.add(node_id, vectors[i])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector.
            top_k: Maximum number of results.
            score_threshold: Minimum similarity threshold.

        Returns:
            List of `(node_id, score)` tuples.
        """
        with self.lock:
            if self.index.ntotal == 0:
                return []

            query_vector = self._normalize(
                query_vector.reshape(1, -1).astype(np.float32)
            )

            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vector, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if score < score_threshold:
                    continue
                node_id = self.idx_to_id.get(idx)
                if node_id is not None:
                    results.append((node_id, float(score)))

            return results

    def remove(self, node_id: int):
        """Mark a vector as removed from the active id mapping."""
        with self.lock:
            if node_id in self.id_to_idx:
                idx = self.id_to_idx.pop(node_id)
                self.idx_to_id.pop(idx, None)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors when using inner-product search."""
        if self.index_type == "IP":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return vectors / norms
        return vectors

    def save(self):
        """Persist the index and id mappings to disk."""
        if not self.storage_path:
            return

        with self.lock:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            index_file = self.storage_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))

            meta_file = self.storage_path / "faiss_meta.pkl"
            meta = {
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "next_idx": self.next_idx,
            }
            with open(meta_file, "wb") as f:
                pickle.dump(meta, f)

    def _load(self):
        """Load a persisted index if one exists."""
        if not self.storage_path:
            return

        index_file = self.storage_path / "faiss.index"
        meta_file = self.storage_path / "faiss_meta.pkl"

        if not index_file.exists() or not meta_file.exists():
            return

        with self.lock:
            self.index = faiss.read_index(str(index_file))

            with open(meta_file, "rb") as f:
                meta = pickle.load(f)

            self.id_to_idx = meta["id_to_idx"]
            self.idx_to_id = meta["idx_to_id"]
            self.next_idx = meta["next_idx"]

    def reset(self):
        """Reset the in-memory and persisted index state."""
        with self.lock:
            if self.index_type == "IP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

            self.id_to_idx.clear()
            self.idx_to_id.clear()
            self.next_idx = 0

            if self.storage_path:
                for f in self.storage_path.glob("faiss*"):
                    f.unlink()

    @property
    def size(self) -> int:
        """Return the number of active vectors."""
        return len(self.id_to_idx)
