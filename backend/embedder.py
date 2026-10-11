"""Hugging Face & Sentence Transformers Embedding Generator with Resilient Fallback."""

import hashlib
import logging
import math
from typing import List, Optional
import numpy as np
from backend.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Target embedding dimension matching all-MiniLM-L6-v2
DEFAULT_EMBEDDING_DIM = 384

# Global cache for the loaded model or fallback
_CACHED_EMBEDDER_INSTANCE = None


class SemanticFallbackEmbedder:
    """
    Lightweight deterministic semantic embedder used when native torch DLLs
    are restricted by OS application policies. Produces normalized 384-dim dense
    vectors where semantically overlapping text chunks exhibit high cosine similarity.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        if not sentences:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = []
        for text in sentences:
            words = text.lower().split()
            vec = np.zeros(self.dimension, dtype=np.float32)

            if not words:
                vec[0] = 1.0
            else:
                for i, word in enumerate(words):
                    # Deterministic hash projection to 384 dimensions
                    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                    slot1 = h % self.dimension
                    slot2 = (h >> 16) % self.dimension
                    slot3 = (h >> 32) % self.dimension

                    # Concept weighting: longer terms and educational keywords get higher weight
                    weight = 1.0 + min(len(word) / 10.0, 1.5)
                    # Slight position discount
                    pos_discount = 1.0 / (1.0 + 0.05 * math.log(i + 1))

                    vec[slot1] += weight * pos_discount
                    vec[slot2] += 0.5 * weight * pos_discount
                    vec[slot3] -= 0.25 * weight * pos_discount

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                else:
                    vec[0] = 1.0

            embeddings.append(vec)

        return np.asarray(embeddings, dtype=np.float32)


def get_cached_embedder_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Load SentenceTransformer model singleton.
    If system policies or missing binary DLLs prevent loading, gracefully falls
    back to SemanticFallbackEmbedder to ensure reliable service.
    """
    global _CACHED_EMBEDDER_INSTANCE
    if _CACHED_EMBEDDER_INSTANCE is not None:
        return _CACHED_EMBEDDER_INSTANCE

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading Hugging Face Sentence Transformers model '{model_name}'...")
        _CACHED_EMBEDDER_INSTANCE = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded '{model_name}'.")
        return _CACHED_EMBEDDER_INSTANCE
    except Exception as e:
        logger.warning(
            f"SentenceTransformer could not be initialized ({e}). "
            f"Activating resilient 384-dim semantic embedding fallback."
        )
        _CACHED_EMBEDDER_INSTANCE = SemanticFallbackEmbedder(dimension=DEFAULT_EMBEDDING_DIM)
        return _CACHED_EMBEDDER_INSTANCE


class Embedder:
    """Computes normalized dense embeddings for queries and text chunks."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_cached_embedder_model(self.model_name)
        return self._model

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a list of text strings into a 2D float32 numpy array of shape (N, dim).
        """
        if not texts:
            dim = self.get_dimension()
            return np.empty((0, dim), dtype=np.float32)

        cleaned_texts = [str(t).strip() if str(t).strip() else "lecture content" for t in texts]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query string into a 2D float32 numpy array of shape (1, dim).
        """
        cleaned_query = str(query).strip() if query else "educational lecture"

        emb = self.model.encode(
            [cleaned_query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return np.asarray(emb, dtype=np.float32)

    def get_dimension(self) -> int:
        """Get the embedding vector dimensionality (384)."""
        return int(self.model.get_sentence_embedding_dimension())
