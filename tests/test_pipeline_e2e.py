"""End-to-End Pipeline integration tests."""

import pytest
from unittest.mock import MagicMock
from backend.pipeline import VideoRetrievalPipeline
from backend.youtube_client import YouTubeClient
from backend.transcript_extractor import TranscriptExtractor


def test_pipeline_with_mocked_transcripts():
    # Setup mock candidate videos
    mock_candidates = [
        {
            "video_id": "test_vid_1",
            "title": "Introduction to Neural Networks",
            "description": "Learn artificial neurons, weights, and biases.",
            "channel_name": "AI Academy",
            "published_at": "2023-01-01T00:00:00Z",
            "thumbnail_url": "https://example.com/thumb1.jpg"
        },
        {
            "video_id": "test_vid_2",
            "title": "Gradient Descent Optimization",
            "description": "How learning rate and gradients update parameters.",
            "channel_name": "Math & ML",
            "published_at": "2023-01-02T00:00:00Z",
            "thumbnail_url": "https://example.com/thumb2.jpg"
        }
    ]

    mock_yt = MagicMock(spec=YouTubeClient)
    mock_yt.search_videos.return_value = mock_candidates

    mock_extractor = MagicMock(spec=TranscriptExtractor)
    mock_extractor.extract_transcript.side_effect = lambda video_id, allow_whisper_fallback=True: [
        {
            "text": "Neural networks consist of layers of interconnected mathematical nodes.",
            "start": 10.0,
            "duration": 5.0
        },
        {
            "text": "Weights and biases determine the strength of connections in learning.",
            "start": 15.0,
            "duration": 6.0
        }
    ] if video_id == "test_vid_1" else [
        {
            "text": "Gradient descent computes the derivative of the cost function with respect to weights.",
            "start": 30.0,
            "duration": 7.0
        },
        {
            "text": "Learning rate controls the step size in parameter updates during backpropagation.",
            "start": 37.0,
            "duration": 8.0
        }
    ]

    pipeline = VideoRetrievalPipeline(
        youtube_client=mock_yt,
        transcript_extractor=mock_extractor
    )

    output = pipeline.process_query("How does gradient descent update weights?", top_k=2)

    assert output["query"] == "How does gradient descent update weights?"
    assert output["total_candidates"] == 2
    assert output["total_results"] > 0
    assert len(output["results"]) <= 2

    top_result = output["results"][0]
    assert "video_id" in top_result
    assert "title" in top_result
    assert "relevance_score" in top_result
    assert "timestamp_url" in top_result
    assert "start_time_formatted" in top_result
    assert "&t=" in top_result["timestamp_url"]


def test_pipeline_empty_query():
    pipeline = VideoRetrievalPipeline()
    output = pipeline.process_query("")
    assert output["total_results"] == 0
    assert output["results"] == []
