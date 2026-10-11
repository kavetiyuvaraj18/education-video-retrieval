"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "embedding_model" in data
    assert data["service"] == "Educational Video Retrieval Backend"


def test_search_empty_query():
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400


def test_search_valid_request_schema():
    # Test that valid schema is accepted and processed
    response = client.post("/search", json={"query": "gradient descent", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "gradient descent"
    assert "results" in data
    assert "execution_time_seconds" in data
