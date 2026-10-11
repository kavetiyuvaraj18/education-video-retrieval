"""Transcript Extraction module with YouTube Captions and Whisper fallback."""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
    VideoUnavailable,
)

from backend.config import (
    CACHE_DIR,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger(__name__)

# Cache for loaded Whisper model instance
_WHISPER_MODEL = None


def get_cached_whisper_model(model_size: str = WHISPER_MODEL_SIZE):
    """Load and cache the Whisper model in memory."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper
            logger.info(f"Loading Whisper model '{model_size}'...")
            _WHISPER_MODEL = whisper.load_model(model_size)
        except Exception as e:
            logger.warning(f"Could not load Whisper model: {e}")
            _WHISPER_MODEL = None
    return _WHISPER_MODEL


class TranscriptExtractor:
    """Extracts timestamped transcript segments from YouTube videos."""

    def __init__(self, cache_dir: Optional[Path] = None, enable_cache: bool = True):
        self.cache_dir = cache_dir or (CACHE_DIR / "transcripts")
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_transcript(
        self,
        video_id: str,
        allow_whisper_fallback: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract timestamped transcript segments for a video.
        Checks cache first, then YouTube captions, then Whisper fallback.

        Args:
            video_id: YouTube video ID.
            allow_whisper_fallback: Whether to attempt audio transcription if captions fail.

        Returns:
            List of segment dicts: [{"text": str, "start": float, "duration": float}]
        """
        if not video_id:
            return []

        # 1. Check local cache
        cached = self._read_cache(video_id)
        if cached is not None:
            logger.info(f"Using cached transcript for video {video_id} ({len(cached)} segments).")
            return cached

        # 2. Try YouTube Captions
        segments = self._fetch_youtube_captions(video_id)
        if segments:
            logger.info(f"Retrieved {len(segments)} caption segments for video {video_id}.")
            self._write_cache(video_id, segments)
            return segments

        # 3. Whisper Audio Fallback
        if allow_whisper_fallback:
            logger.info(f"Captions unavailable for {video_id}. Attempting Whisper transcription fallback...")
            segments = self._transcribe_with_whisper(video_id)
            if segments:
                logger.info(f"Whisper successfully transcribed {len(segments)} segments for video {video_id}.")
                self._write_cache(video_id, segments)
                return segments

        logger.warning(f"Failed to obtain transcript for video {video_id}.")
        return []

    def _fetch_youtube_captions(self, video_id: str) -> List[Dict[str, Any]]:
        """Fetch official or auto-generated YouTube captions."""
        try:
            # Check for modern youtube-transcript-api v1.2+
            api_instance = YouTubeTranscriptApi() if callable(YouTubeTranscriptApi) else None
            
            # 1. Try v1.2+ instance fetch method
            if api_instance and hasattr(api_instance, "fetch"):
                try:
                    fetched = api_instance.fetch(video_id, languages=["en", "en-US", "en-GB"])
                    raw_snippets = getattr(fetched, "snippets", fetched)
                    return self._normalize_segments(raw_snippets)
                except Exception as e:
                    logger.info(f"Direct fetch for {video_id} failed ({e}); trying list...")

            # 2. Try v1.2+ instance list method or legacy class list_transcripts
            transcript_list = None
            if api_instance and hasattr(api_instance, "list"):
                try:
                    transcript_list = api_instance.list(video_id)
                except Exception:
                    transcript_list = None
            elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                except Exception:
                    transcript_list = None

            if transcript_list:
                for t in transcript_list:
                    try:
                        fetched = t.fetch()
                        raw_snippets = getattr(fetched, "snippets", fetched)
                        return self._normalize_segments(raw_snippets)
                    except Exception:
                        continue

            # 3. Fallback to legacy class method get_transcript
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                raw_segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
                return self._normalize_segments(raw_segments)

            return []

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript) as e:
            logger.info(f"Captions not available for {video_id}: {type(e).__name__}")
            return []
        except Exception as e:
            logger.warning(f"Error fetching YouTube captions for {video_id}: {e}")
            return []

    def _transcribe_with_whisper(self, video_id: str) -> List[Dict[str, Any]]:
        """Download audio with yt-dlp and transcribe using Whisper."""
        try:
            import yt_dlp
        except ImportError:
            logger.warning("yt-dlp not installed; cannot perform Whisper fallback.")
            return []

        whisper_model = get_cached_whisper_model()
        if whisper_model is None:
            logger.warning("Whisper model not available.")
            return []

        temp_dir = tempfile.mkdtemp(prefix=f"whisper_{video_id}_")
        audio_path = os.path.join(temp_dir, f"{video_id}.mp3")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(temp_dir, f"{video_id}.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
            # Limit download to first 10 minutes to save processing time during retrieval
            "download_ranges": lambda info_dict, ydl: [{"start_time": 0, "end_time": 600}],
        }

        try:
            video_url = f"https://www.youtube.com/watch?v=api_{video_id}" if video_id.startswith("mock_") else f"https://www.youtube.com/watch?v={video_id}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            if not os.path.exists(audio_path):
                # Check for other audio extensions
                found = list(Path(temp_dir).glob(f"{video_id}.*"))
                if found:
                    audio_path = str(found[0])
                else:
                    logger.warning(f"Audio file for {video_id} could not be downloaded.")
                    return []

            # Transcribe audio file
            result = whisper_model.transcribe(audio_path, fp16=False)
            raw_segments = result.get("segments", [])
            segments: List[Dict[str, Any]] = []
            for seg in raw_segments:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start + 2.0))
                text = str(seg.get("text", "")).strip()
                if text:
                    segments.append({
                        "text": text,
                        "start": round(start, 2),
                        "duration": round(max(0.1, end - start), 2)
                    })
            return segments

        except Exception as e:
            logger.warning(f"Whisper transcription failed for {video_id}: {e}")
            return []
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _normalize_segments(self, raw_segments: List[Any]) -> List[Dict[str, Any]]:
        """Normalize segment timing and text formatting."""
        normalized: List[Dict[str, Any]] = []
        for item in raw_segments:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                start = float(item.get("start", 0.0))
                duration = float(item.get("duration", 0.0))
            else:
                text = str(getattr(item, "text", "")).strip()
                start = float(getattr(item, "start", 0.0))
                duration = float(getattr(item, "duration", 0.0))

            if not text:
                continue

            normalized.append({
                "text": text,
                "start": round(start, 2),
                "duration": round(max(0.1, duration), 2)
            })
        return normalized

    def _read_cache(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        if not self.enable_cache:
            return None
        cache_file = self.cache_dir / f"{video_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read transcript cache for {video_id}: {e}")
        return None

    def _write_cache(self, video_id: str, segments: List[Dict[str, Any]]) -> None:
        if not self.enable_cache or not segments:
            return
        cache_file = self.cache_dir / f"{video_id}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write transcript cache for {video_id}: {e}")
