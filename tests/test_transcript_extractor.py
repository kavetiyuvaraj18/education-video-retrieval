"""Tests for transcript cache behavior and caption-first extraction."""

import json
import time

from backend.transcript_extractor import TranscriptExtractor


def test_successful_transcript_is_reused_from_cache(tmp_path, monkeypatch):
    extractor = TranscriptExtractor(cache_dir=tmp_path)
    captions = [{"text": "A caption", "start": 2.0, "duration": 1.5}]
    fetch = []
    monkeypatch.setattr(
        extractor,
        "_fetch_youtube_captions",
        lambda video_id: fetch.append(video_id) or captions,
    )
    monkeypatch.setattr(
        extractor,
        "_transcribe_with_whisper",
        lambda video_id: (_ for _ in ()).throw(AssertionError("Whisper should not run")),
    )

    assert extractor.extract_transcript("video-a") == captions
    assert extractor.extract_transcript("video-a") == captions
    assert fetch == ["video-a"]


def test_failed_transcript_is_cached_until_expiry(tmp_path, monkeypatch):
    extractor = TranscriptExtractor(
        cache_dir=tmp_path,
        failed_cache_ttl_seconds=60,
    )
    caption_calls = []
    whisper_calls = []
    monkeypatch.setattr(
        extractor,
        "_fetch_youtube_captions",
        lambda video_id: caption_calls.append(video_id) or [],
    )
    monkeypatch.setattr(
        extractor,
        "_transcribe_with_whisper",
        lambda video_id: whisper_calls.append(video_id) or [],
    )

    assert extractor.extract_transcript("video-b") == []
    assert extractor.extract_transcript("video-b") == []
    assert caption_calls == ["video-b"]
    assert whisper_calls == ["video-b"]

    failure_cache = tmp_path / "video-b.failure.json"
    entry = json.loads(failure_cache.read_text(encoding="utf-8"))
    entry["expires_at"] = time.time() - 1
    failure_cache.write_text(json.dumps(entry), encoding="utf-8")

    assert extractor.extract_transcript("video-b") == []
    assert caption_calls == ["video-b", "video-b"]
    assert whisper_calls == ["video-b", "video-b"]


def test_cached_failure_without_whisper_retries_when_whisper_is_enabled(
    tmp_path, monkeypatch
):
    extractor = TranscriptExtractor(
        cache_dir=tmp_path,
        failed_cache_ttl_seconds=60,
    )
    caption_calls = []
    whisper_calls = []
    monkeypatch.setattr(
        extractor,
        "_fetch_youtube_captions",
        lambda video_id: caption_calls.append(video_id) or [],
    )
    monkeypatch.setattr(
        extractor,
        "_transcribe_with_whisper",
        lambda video_id: whisper_calls.append(video_id) or [],
    )

    assert extractor.extract_transcript("video-c", allow_whisper_fallback=False) == []
    assert extractor.extract_transcript("video-c", allow_whisper_fallback=True) == []
    assert caption_calls == ["video-c"]
    assert whisper_calls == ["video-c"]
