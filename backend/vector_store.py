"""FAISS Dense Vector Indexing and Semantic Retrieval Module."""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS Vector Store supporting Cosine Similarity via Normalized Inner Product."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = None
        self._chunks: List[Dict[str, Any]] = []
        self._init_index()

    def _init_index(self):
        """Initialize FAISS Flat Inner Product index."""
        try:
            import faiss
            # IndexFlatIP computes inner products. When vectors are L2-normalized,
            # inner product is equal to cosine similarity.
            self._index = faiss.IndexFlatIP(self.dimension)
        except ImportError as e:
            logger.error("faiss is not installed. Please install faiss-cpu.")
            raise RuntimeError("FAISS library not installed.") from e

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors currently indexed."""
        return self._index.ntotal if self._index is not None else 0

    def add_chunks(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Add chunk embeddings and corresponding metadata to the FAISS index.

        Args:
            embeddings: 2D numpy array of shape (N, dimension), dtype float32.
            chunks: List of N chunk dictionaries corresponding to each vector.
        """
        if embeddings.shape[0] == 0 or len(chunks) == 0:
            logger.warning("Empty embeddings or chunks provided to FAISS index.")
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch between embeddings count ({embeddings.shape[0]}) and chunks count ({len(chunks)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match FAISS index dimension ({self.dimension})"
            )

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Normalize if not already normalized (safeguard)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms

        self._index.add(normalized_vectors)
        self._chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors to FAISS index. Total indexed: {self.total_vectors}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search FAISS index for nearest neighbors.

        Args:
            query_embedding: 2D numpy array of shape (1, dimension), dtype float32.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of dictionaries containing chunk data and 'similarity_score'.
        """
        if self.total_vectors == 0:
            logger.warning("Search attempted on empty FAISS index.")
            return []

        k = min(top_k, self.total_vectors)
        q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)

        # Ensure query is 2D
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        # Normalize query vector
        norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        q_vec = q_vec / norm

        # Search index
        scores, indices = self._index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk_data = dict(self._chunks[idx])
            chunk_data["similarity_score"] = float(score)
            results.append(chunk_data)

        return results

    def reset(self) -> None:
        """Reset index and clear metadata."""
        self._init_index()
        self._chunks.clear()
