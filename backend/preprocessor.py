"""Transcript Preprocessing and Overlapping Timestamped Chunking Module."""

import re
import logging
from typing import List, Dict, Any, Optional
from backend.config import CHUNK_WORD_SIZE, CHUNK_OVERLAP_WORDS

logger = logging.getLogger(__name__)

# Patterns to clean from transcripts (e.g., [Music], [Applause], &amp;)
NOISE_PATTERNS = [
    re.compile(r"\[\s*(music|applause|laughter|cheering|inaudible)\s*\]", re.IGNORECASE),
    re.compile(r"\(\s*(music|applause|laughter|cheering|inaudible)\s*\)", re.IGNORECASE),
    re.compile(r"&amp;", re.IGNORECASE),
    re.compile(r"&#39;", re.IGNORECASE),
    re.compile(r"&quot;", re.IGNORECASE),
    re.compile(r"\s+", re.UNICODE),
]


def clean_text(text: str) -> str:
    """
    Clean transcript text by removing noise tokens and normalizing whitespace,
    while carefully preserving technical jargon, numbers, and symbols.
    """
    if not text:
        return ""

    cleaned = text
    # Replace noise tags with a space
    for pat in NOISE_PATTERNS[:-1]:
        cleaned = pat.sub(" ", cleaned)

    # Normalize multiple whitespace characters to a single space
    cleaned = NOISE_PATTERNS[-1].sub(" ", cleaned).strip()
    return cleaned


class TranscriptChunker:
    """Chunks transcript segments into overlapping windows preserving timestamps."""

    def __init__(
        self,
        chunk_size: int = CHUNK_WORD_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP_WORDS
    ):
        self.chunk_size = max(10, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    def create_chunks(
        self,
        video_id: str,
        segments: List[Dict[str, Any]],
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from transcript segments.

        Args:
            video_id: Unique YouTube video ID.
            segments: List of segments with 'text', 'start', 'duration'.
            video_metadata: Optional metadata (title, channel, etc.) to attach to each chunk.

        Returns:
            List of chunk dicts:
            {
                "chunk_id": str,
                "video_id": str,
                "text": str,
                "start_time": float,
                "end_time": float,
                "start_formatted": str,
                "end_formatted": str,
                "metadata": dict
            }
        """
        if not segments:
            return []

        # Filter and clean segments
        valid_segments = []
        for seg in segments:
            raw_text = seg.get("text", "")
            text = clean_text(raw_text)
            if text:
                valid_segments.append({
                    "text": text,
                    "start": float(seg.get("start", 0.0)),
                    "duration": float(seg.get("duration", 0.0)),
                    "end": float(seg.get("start", 0.0)) + float(seg.get("duration", 0.0))
                })

        if not valid_segments:
            return []

        # Break into words with word-level timing estimates
        timed_words = []
        for seg in valid_segments:
            words = seg["text"].split()
            if not words:
                continue
            seg_start = seg["start"]
            seg_end = max(seg["end"], seg_start + 0.1)
            time_per_word = (seg_end - seg_start) / len(words)

            for i, w in enumerate(words):
                w_start = seg_start + (i * time_per_word)
                w_end = w_start + time_per_word
                timed_words.append({
                    "word": w,
                    "start": w_start,
                    "end": w_end
                })

        if not timed_words:
            return []

        chunks: List[Dict[str, Any]] = []
        step = self.chunk_size - self.chunk_overlap
        total_words = len(timed_words)

        chunk_idx = 0
        for start_idx in range(0, total_words, step):
            end_idx = min(start_idx + self.chunk_size, total_words)
            window = timed_words[start_idx:end_idx]

            if not window:
                continue

            chunk_text = " ".join(item["word"] for item in window).strip()
            if not chunk_text:
                continue

            chunk_start = round(window[0]["start"], 2)
            chunk_end = round(window[-1]["end"], 2)
            if chunk_end <= chunk_start:
                chunk_end = round(chunk_start + 1.0, 2)

            chunk_id = f"{video_id}_chunk_{chunk_idx:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "video_id": video_id,
                "text": chunk_text,
                "start_time": chunk_start,
                "end_time": chunk_end,
                "start_formatted": self.format_timestamp(chunk_start),
                "end_formatted": self.format_timestamp(chunk_end),
                "metadata": video_metadata or {}
            })
            chunk_idx += 1

            # If the current window reached the end of the text, stop
            if end_idx >= total_words:
                break

        return chunks

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS."""
        secs = max(0, int(round(seconds)))
        hrs = secs // 3600
        mins = (secs % 3600) // 60
        remaining_secs = secs % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{remaining_secs:02d}"
        return f"{mins:02d}:{remaining_secs:02d}"
