"""Tests for Video Ranker and Timestamp Generator."""

import pytest
from backend.ranker import VideoRanker
from backend.timestamp_generator import TimestampGenerator, format_seconds, generate_youtube_url


def test_video_ranker_deduplication_and_ordering():
    ranker = VideoRanker(top_k=2)

    chunks = [
        {"video_id": "vid_A", "chunk_id": "c1", "similarity_score": 0.85, "text": "backprop step 1"},
        {"video_id": "vid_A", "chunk_id": "c2", "similarity_score": 0.80, "text": "backprop step 2"},
        {"video_id": "vid_B", "chunk_id": "c3", "similarity_score": 0.92, "text": "best match on B"},
        {"video_id": "vid_C", "chunk_id": "c4", "similarity_score": 0.65, "text": "weak match on C"},
    ]

    metadata = {
        "vid_A": {"title": "Calculus of Backpropagation", "channel_name": "3Blue1Brown"},
        "vid_B": {"title": "Neural Networks Demystified", "channel_name": "Welch Labs"},
        "vid_C": {"title": "Linear Regression Basics", "channel_name": "StatQuest"},
    }

    ranked = ranker.rank_videos(chunks, metadata)

    # Should deduplicate to at most 2 videos
    assert len(ranked) == 2
    # vid_B had 0.92 vs vid_A had 0.85 + bonus (~0.89), so vid_B should be first
    assert ranked[0]["video_id"] == "vid_B"
    assert ranked[1]["video_id"] == "vid_A"
    assert ranked[0]["title"] == "Neural Networks Demystified"


def test_timestamp_generator():
    gen = TimestampGenerator()
    video_item = {
        "video_id": "aircAruvnKk",
        "relevance_score": 0.88,
        "best_chunk": {
            "chunk_id": "aircAruvnKk_chunk_0001",
            "text": "A neuron simply holds a number between 0 and 1.",
            "start_time": 125.4,
            "end_time": 155.0
        }
    }

    enriched = gen.enrich_video_result(video_item)
    assert enriched["start_time_seconds"] == 125.4
    assert enriched["start_time_formatted"] == "02:05"
    assert enriched["end_time_formatted"] == "02:35"
    assert "https://www.youtube.com/watch?v=aircAruvnKk&t=125s" == enriched["timestamp_url"]
    assert "A neuron simply holds a number" in enriched["topic_text"]


def test_format_helpers():
    assert format_seconds(0) == "00:00"
    assert format_seconds(65) == "01:05"
    assert format_seconds(3605) == "01:00:05"
    assert generate_youtube_url("xyz123", 90.7) == "https://www.youtube.com/watch?v=xyz123&t=90s"
