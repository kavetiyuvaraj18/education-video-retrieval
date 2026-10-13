"""End-to-End Pipeline integration tests."""

import pytest
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
from backend.pipeline import VideoRetrievalPipeline
from backend.youtube_client import YouTubeClient
from backend.transcript_extractor import TranscriptExtractor
from backend.preprocessor import TranscriptChunker


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


class StubEmbedder:
    def __init__(self):
        self.text_encode_calls = 0

    @property
    def cache_identity(self):
        return "stub-embedder-v1"

    def encode_texts(self, texts, normalize=True):
        self.text_encode_calls += 1
        return np.zeros((len(texts), 2), dtype=np.float32)

    def encode_query(self, query, normalize=True):
        return np.zeros((1, 2), dtype=np.float32)

    def get_dimension(self):
        return 2


@pytest.mark.parametrize("requested_count", [3, 6, 8])
@pytest.mark.parametrize("insufficient_candidates", [False, True])
def test_pipeline_returns_requested_unique_video_count_when_available(
    monkeypatch, tmp_path, requested_count, insufficient_candidates
):
    monkeypatch.setattr("backend.pipeline.CACHE_DIR", tmp_path / "cache")
    available_count = requested_count - 1 if insufficient_candidates else requested_count
    candidates = [
        {
            "video_id": f"video_{index}",
            "title": f"Lecture {index}",
            "channel_name": "Test channel",
            "thumbnail_url": f"https://example.com/{index}.jpg",
        }
        for index in range(available_count)
    ]
    all_chunks = []
    for candidate in candidates:
        count = 31 if candidate["video_id"] == "video_0" else 1
        for index in range(count):
            all_chunks.append(
                {
                    "chunk_id": f'{candidate["video_id"]}_{index}',
                    "video_id": candidate["video_id"],
                    "text": f'Transcript chunk {candidate["video_id"]} {index}',
                    "start_time": float(index),
                    "end_time": float(index + 1),
                    "metadata": candidate,
                    "similarity_score": 1.0 - (len(all_chunks) * 0.001),
                }
            )

    search_limits = []
    index_builds = []

    class StubFAISSIndex:
        def __init__(self, dimension):
            self.chunks = []
            index_builds.append(dimension)

        def add_chunks(self, embeddings, chunks):
            self.chunks = chunks

        def search(self, query_embedding, top_k):
            search_limits.append(top_k)
            return sorted(
                self.chunks,
                key=lambda chunk: chunk["similarity_score"],
                reverse=True,
            )[:top_k]

    monkeypatch.setattr("backend.pipeline.FAISSIndex", StubFAISSIndex)

    youtube_client = MagicMock(spec=YouTubeClient)
    youtube_client.search_videos.return_value = candidates
    transcript_extractor = MagicMock(spec=TranscriptExtractor)
    transcript_extractor.extract_transcript.return_value = [
        {"text": "test transcript", "start": 0.0, "duration": 1.0}
    ]
    chunker = TranscriptChunker()
    chunker.create_chunks = lambda video_id, segments, video_metadata: [
        chunk for chunk in all_chunks if chunk["video_id"] == video_id
    ]
    embedder = StubEmbedder()
    pipeline = VideoRetrievalPipeline(
        youtube_client=youtube_client,
        transcript_extractor=transcript_extractor,
        chunker=chunker,
        embedder=embedder,
        db=MagicMock(),
    )

    output = pipeline.process_query("test query", top_k=requested_count)
    repeated_output = pipeline.process_query("test query", top_k=requested_count)
    result_ids = [result["video_id"] for result in output["results"]]
    expected_count = available_count

    assert output["total_results"] == expected_count
    assert len(result_ids) == expected_count
    assert len(set(result_ids)) == expected_count
    assert repeated_output["results"] == output["results"]
    assert search_limits[0] == 30
    assert search_limits[-1] == len(all_chunks)
    assert embedder.text_encode_calls == 1
    assert len(index_builds) == 1
    fallback_flags = [
        call.kwargs["allow_whisper_fallback"]
        for call in transcript_extractor.extract_transcript.call_args_list[:available_count]
    ]
    assert fallback_flags == [
        index < 2 for index in range(available_count)
    ]


def test_pipeline_keeps_results_when_one_transcript_fetch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.pipeline.CACHE_DIR", tmp_path / "cache")
    candidates = [
        {
            "video_id": "broken-video",
            "title": "Unavailable captions",
            "channel_name": "Test",
            "thumbnail_url": "",
        },
        {
            "video_id": "working-video",
            "title": "Available captions",
            "channel_name": "Test",
            "thumbnail_url": "",
        },
    ]
    youtube_client = MagicMock(spec=YouTubeClient)
    youtube_client.search_videos.return_value = candidates
    transcript_extractor = MagicMock(spec=TranscriptExtractor)

    def extract(video_id, allow_whisper_fallback=True):
        if video_id == "broken-video":
            raise RuntimeError("caption provider unavailable")
        return [{"text": "A useful lecture topic", "start": 14.0, "duration": 4.0}]

    transcript_extractor.extract_transcript.side_effect = extract
    pipeline = VideoRetrievalPipeline(
        youtube_client=youtube_client,
        transcript_extractor=transcript_extractor,
        embedder=StubEmbedder(),
        db=MagicMock(),
    )

    output = pipeline.process_query("useful lecture", top_k=3)

    assert output["total_results"] == 1
    assert [item["video_id"] for item in output["results"]] == ["working-video"]


def test_concurrent_searches_encode_same_cache_miss_once(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.pipeline.CACHE_DIR", tmp_path / "cache")
    candidate = {
        "video_id": "shared-video",
        "title": "Shared lecture",
        "channel_name": "Test",
        "thumbnail_url": "",
    }
    youtube_client = MagicMock(spec=YouTubeClient)
    youtube_client.search_videos.return_value = [candidate]
    transcript_barrier = threading.Barrier(2)

    class ConcurrentTranscriptExtractor:
        def extract_transcript(self, video_id, allow_whisper_fallback=True):
            transcript_barrier.wait(timeout=5)
            return [{"text": "A shared useful lecture topic", "start": 3.0, "duration": 4.0}]

    class BlockingEmbedder(StubEmbedder):
        def __init__(self):
            super().__init__()
            self.encoding_started = threading.Event()
            self.release_encoding = threading.Event()
            self._calls_lock = threading.Lock()

        def encode_texts(self, texts, normalize=True):
            with self._calls_lock:
                self.text_encode_calls += 1
            self.encoding_started.set()
            if not self.release_encoding.wait(timeout=5):
                raise TimeoutError("test did not release embedding")
            return np.zeros((len(texts), 2), dtype=np.float32)

    class OneChunker:
        chunk_size = 60
        chunk_overlap = 15

        def create_chunks(self, video_id, segments, video_metadata):
            return [{
                "chunk_id": f"{video_id}_chunk",
                "video_id": video_id,
                "text": segments[0]["text"],
                "start_time": segments[0]["start"],
                "end_time": segments[0]["start"] + segments[0]["duration"],
                "metadata": video_metadata,
            }]

    embedder = BlockingEmbedder()
    pipeline = VideoRetrievalPipeline(
        youtube_client=youtube_client,
        transcript_extractor=ConcurrentTranscriptExtractor(),
        chunker=OneChunker(),
        embedder=embedder,
        db=MagicMock(),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        searches = [
            executor.submit(pipeline.process_query, "shared topic", top_k=1)
            for _ in range(2)
        ]
        assert embedder.encoding_started.wait(timeout=5)
        time.sleep(0.1)
        embedder.release_encoding.set()
        outputs = [search.result(timeout=5) for search in searches]

    assert embedder.text_encode_calls == 1
    assert all(output["total_results"] == 1 for output in outputs)
