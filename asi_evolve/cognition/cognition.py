"""Cognition-store implementation."""

import json
import uuid
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

from ..utils.structures import CognitionItem
from ..database.faiss_index import FAISSIndex
from ..database.embedding import EmbeddingService


class Cognition:
    """
    Persistent cognition store with embedding-based retrieval.

    Features:
    - CRUD operations for cognition items
    - Semantic retrieval through FAISS
    - Local persistence to disk
    """

    def __init__(
        self,
        storage_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        retrieval_top_k: int = 5,
        score_threshold: float = 0.5,
        faiss_index_type: str = "IP",
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.lock = RLock()
        self.retrieval_top_k = retrieval_top_k
        self.score_threshold = score_threshold

        self.items: Dict[str, CognitionItem] = {}

        self.embedding = EmbeddingService(model_name=embedding_model)

        self.faiss = FAISSIndex(
            dimension=embedding_dim,
            index_type=faiss_index_type,
            storage_path=self.storage_dir / "faiss",
        )

        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}
        self.next_int_id = 0

        self._load()

    def _get_int_id(self, str_id: str) -> int:
        if str_id not in self.str_to_int:
            self.str_to_int[str_id] = self.next_int_id
            self.int_to_str[self.next_int_id] = str_id
            self.next_int_id += 1
        return self.str_to_int[str_id]

    def add(self, item: CognitionItem) -> str:
        with self.lock:
            if not item.id:
                item.id = str(uuid.uuid4())

            self.items[item.id] = item

            if item.content:
                int_id = self._get_int_id(item.id)
                vector = self.embedding.encode(item.content)
                self.faiss.add(int_id, vector)

            self._save()
            return item.id

    def add_batch(self, items: List[CognitionItem]) -> List[str]:
        return [self.add(item) for item in items]

    def remove(self, item_id: str) -> bool:
        with self.lock:
            if item_id not in self.items:
                return False

            del self.items[item_id]

            if item_id in self.str_to_int:
                int_id = self.str_to_int.pop(item_id)
                self.int_to_str.pop(int_id, None)
                self.faiss.remove(int_id)

            self._save()
            return True

    def remove_batch(self, item_ids: List[str]) -> int:
        return sum(1 for iid in item_ids if self.remove(iid))

    def get(self, item_id: str) -> Optional[CognitionItem]:
        return self.items.get(item_id)

    def get_all(self) -> List[CognitionItem]:
        return list(self.items.values())

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[CognitionItem, float]]:
        top_k = top_k or self.retrieval_top_k
        score_threshold = score_threshold if score_threshold is not None else self.score_threshold

        query_vector = self.embedding.encode(query)
        results = self.faiss.search(query_vector, top_k, score_threshold)

        items_with_scores = []
        for int_id, score in results:
            str_id = self.int_to_str.get(int_id)
            if str_id:
                item = self.items.get(str_id)
                if item:
                    items_with_scores.append((item, score))

        return items_with_scores

    def search(self, query: str, top_k: Optional[int] = None) -> List[CognitionItem]:
        results = self.retrieve(query, top_k)
        return [item for item, _ in results]

    def reset(self):
        with self.lock:
            self.items.clear()
            self.str_to_int.clear()
            self.int_to_str.clear()
            self.next_int_id = 0
            self.faiss.reset()

            data_file = self.storage_dir / "cognition.json"
            if data_file.exists():
                data_file.unlink()

    def _save(self):
        data_file = self.storage_dir / "cognition.json"

        data = {
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "str_to_int": self.str_to_int,
            "int_to_str": {str(k): v for k, v in self.int_to_str.items()},
            "next_int_id": self.next_int_id,
        }

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.faiss.save()

    def _load(self):
        data_file = self.storage_dir / "cognition.json"

        if not data_file.exists():
            return

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item_id, item_data in data.get("items", {}).items():
            item = CognitionItem.from_dict(item_data)
            self.items[item_id] = item

        self.str_to_int = data.get("str_to_int", {})
        self.int_to_str = {int(k): v for k, v in data.get("int_to_str", {}).items()}
        self.next_int_id = data.get("next_int_id", 0)

    @property
    def size(self) -> int:
        return len(self.items)

    def __len__(self) -> int:
        return self.size
