"""Timestamp and Topic Link Generation Module."""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    secs = max(0, int(round(seconds)))
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    rem_secs = secs % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{rem_secs:02d}"
    return f"{mins:02d}:{rem_secs:02d}"


def generate_youtube_url(video_id: str, start_seconds: float) -> str:
    """Generate a direct YouTube link with start timestamp parameter."""
    t_seconds = max(0, int(start_seconds))
    return f"https://www.youtube.com/watch?v={video_id}&t={t_seconds}s"


class TimestampGenerator:
    """Generates structured timestamp evidence and direct video links."""

    def enrich_video_result(self, video_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a ranked video result with formatted timestamps, topic excerpt,
        and playable YouTube timestamp link.

        Args:
            video_result: Dictionary from VideoRanker containing 'video_id', 'best_chunk', etc.

        Returns:
            Enriched dictionary ready for frontend consumption.
        """
        video_id = video_result.get("video_id", "")
        best_chunk = video_result.get("best_chunk", {})

        start_time = float(best_chunk.get("start_time", 0.0))
        end_time = float(best_chunk.get("end_time", start_time + 30.0))

        if end_time <= start_time:
            end_time = start_time + 10.0

        # Construct topic summary text from best chunk
        topic_text = best_chunk.get("text", "").strip()

        # Generate URLs
        clean_video_url = f"https://www.youtube.com/watch?v={video_id}"
        timestamp_url = generate_youtube_url(video_id, start_time)

        # Enrich the dict
        enriched = dict(video_result)
        enriched.update({
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "start_time_formatted": format_seconds(start_time),
            "end_time_formatted": format_seconds(end_time),
            "topic_text": topic_text,
            "video_url": clean_video_url,
            "timestamp_url": timestamp_url
        })

        return enriched

    def enrich_all(self, ranked_videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich a list of ranked video results."""
        return [self.enrich_video_result(v) for v in ranked_videos]
