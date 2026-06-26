"""Embedding service wrapper."""

from typing import List, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class EmbeddingService:
    """Local embedding service backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Args:
            model_name: Embedding model name.
            device: Runtime device (`cpu` or `cuda`).
        """
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text into embedding vectors.

        Args:
            texts: A single string or a list of strings.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            Array of shape `(n, dimension)`.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimension
