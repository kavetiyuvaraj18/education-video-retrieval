"""Tests for YouTube Client module."""

import pytest
from backend.youtube_client import YouTubeClient, YouTubeRetrievalError


def test_youtube_client_empty_query():
    client = YouTubeClient(api_key="dummy_key")
    results = client.search_videos("")
    assert results == []

    results_whitespace = client.search_videos("   ")
    assert results_whitespace == []


def test_youtube_client_fallback_candidates():
    # Client without API key should fall back to curated educational candidates
    client = YouTubeClient(api_key="")
    results = client.search_videos("gradient descent neural networks", max_results=6)
    assert len(results) > 0
    assert len(results) <= 6

    first = results[0]
    assert "video_id" in first
    assert "title" in first
    assert "channel_name" in first
    assert "thumbnail_url" in first


def test_youtube_client_disallow_fallback():
    client = YouTubeClient(api_key="")
    with pytest.raises(YouTubeRetrievalError):
        client.search_videos("machine learning", allow_fallback=False)


def test_youtube_client_result_limit():
    client = YouTubeClient(api_key="")
    results = client.search_videos("backpropagation", max_results=2)
    assert len(results) <= 2
