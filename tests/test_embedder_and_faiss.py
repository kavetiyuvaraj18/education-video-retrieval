"""Tests for Embedder and FAISS Vector Store."""

import pytest
import numpy as np
from backend.vector_store import FAISSIndex


def test_faiss_index_basic_operations():
    dimension = 4
    index = FAISSIndex(dimension=dimension)
    assert index.total_vectors == 0

    # Add mock vectors
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    chunks = [
        {"chunk_id": "c1", "video_id": "v1", "text": "Deep learning intro"},
        {"chunk_id": "c2", "video_id": "v2", "text": "Gradient descent details"},
        {"chunk_id": "c3", "video_id": "v3", "text": "Backpropagation algorithm"}
    ]

    index.add_chunks(embeddings, chunks)
    assert index.total_vectors == 3

    # Query vector close to c2
    query = np.array([[0.1, 0.9, 0.0, 0.0]], dtype=np.float32)
    results = index.search(query, top_k=2)

    assert len(results) == 2
    # c2 should have highest cosine similarity
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["similarity_score"] > 0.9


def test_faiss_empty_search():
    index = FAISSIndex(dimension=4)
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert index.search(query, top_k=5) == []


def test_faiss_dimension_mismatch():
    index = FAISSIndex(dimension=4)
    bad_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        index.add_chunks(bad_embeddings, [{"chunk_id": "c1"}])
