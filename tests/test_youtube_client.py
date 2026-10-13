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
    """When allow_fallback=False and no API key + yt-dlp returns nothing, raise YouTubeRetrievalError."""
    from unittest.mock import patch
    client = YouTubeClient(api_key="")
    # Patch yt-dlp live search to simulate failure (e.g. network unavailable)
    with patch.object(client, "_search_with_ytdlp", return_value=[]):
        with pytest.raises(YouTubeRetrievalError):
            client.search_videos("machine learning", allow_fallback=False)


def test_youtube_client_result_limit():
    client = YouTubeClient(api_key="")
    results = client.search_videos("backpropagation", max_results=2)
    assert len(results) <= 2


def test_youtube_client_reuses_recent_candidate_search(monkeypatch):
    client = YouTubeClient(api_key="")
    calls = []
    videos = [{"video_id": "video_1", "title": "Lecture"}]
    monkeypatch.setattr(
        client,
        "_search_with_ytdlp",
        lambda query, limit: calls.append((query, limit)) or videos,
    )

    first = client.search_videos("  Neural Networks  ", max_results=2)
    repeated = client.search_videos("neural networks", max_results=2)

    assert first == repeated
    assert first is not repeated
    assert calls == [("Neural Networks", 2)]
