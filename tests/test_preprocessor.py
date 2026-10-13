"""Tests for Transcript Preprocessing and Chunking."""

import pytest
from backend.preprocessor import clean_text, TranscriptChunker


def test_clean_text():
    # Remove noise tags like [Music] while preserving text and technical terms
    noisy = "Hello world! [Music] Today we study backpropagation &amp; gradient descent. [Applause]"
    cleaned = clean_text(noisy)
    assert "[Music]" not in cleaned
    assert "[Applause]" not in cleaned
    assert "&amp;" not in cleaned
    assert "backpropagation" in cleaned
    assert "gradient descent" in cleaned


def test_chunker_empty_input():
    chunker = TranscriptChunker(chunk_size=50, chunk_overlap=10)
    assert chunker.create_chunks("vid_1", []) == []


def test_chunker_valid_segments():
    chunker = TranscriptChunker(chunk_size=10, chunk_overlap=2)
    segments = [
        {"text": "In this lecture we define supervised learning", "start": 0.0, "duration": 5.0},
        {"text": "where we have inputs and known corresponding output labels.", "start": 5.0, "duration": 6.0},
        {"text": "The goal is to minimize the empirical risk or loss function.", "start": 11.0, "duration": 5.0},
        {"text": "Gradient descent computes the gradients of the parameters.", "start": 16.0, "duration": 6.0},
    ]

    chunks = chunker.create_chunks("vid_1", segments, {"title": "ML Intro"})
    assert len(chunks) > 0

    for i, chunk in enumerate(chunks):
        assert chunk["video_id"] == "vid_1"
        assert chunk["chunk_id"] == f"vid_1_chunk_{i:04d}"
        assert chunk["start_time"] < chunk["end_time"]
        assert len(chunk["text"].split()) <= 10
        assert "start_formatted" in chunk
        assert "end_formatted" in chunk


def test_format_timestamp():
    assert TranscriptChunker.format_timestamp(45) == "00:45"
    assert TranscriptChunker.format_timestamp(125) == "02:05"
    assert TranscriptChunker.format_timestamp(3665) == "01:01:05"


def test_chunk_timestamps_preserve_caption_segment_boundaries():
    chunker = TranscriptChunker(chunk_size=10, chunk_overlap=0)
    chunks = chunker.create_chunks(
        "timed_video",
        [
            {"text": "A caption with several words", "start": 12.25, "duration": 3.5},
            {"text": "another caption interval", "start": 20.0, "duration": 2.0},
        ],
    )

    assert chunks[0]["start_time"] == 12.25
    assert chunks[0]["end_time"] == 22.0
