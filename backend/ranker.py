"""Video Ranking and Selection Module."""

import logging
from typing import List, Dict, Any, Optional
from backend.config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


class VideoRanker:
    """Aggregates chunk retrieval scores to rank distinct videos."""

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k

    def rank_videos(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        candidate_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank videos based on retrieved transcript chunks.

        Aggregation strategy:
        - Group chunks by `video_id`.
        - For each video, the primary score is the maximum similarity score of its chunks.
        - To reward videos with multiple high-quality matches without allowing repetitive
          chunks to dominate, add a discounted secondary bonus:
          video_score = max_score + 0.1 * avg(other_top_scores)
        - Pick the single strongest matching chunk (or top group) as the representative evidence.
        - Return up to `top_k` unique videos sorted descending by final score.

        Args:
            retrieved_chunks: List of chunk dictionaries containing 'video_id' and 'similarity_score'.
            candidate_metadata_map: Dict mapping video_id -> full video metadata (title, channel, etc.).

        Returns:
            List of ranked video result dictionaries.
        """
        if not retrieved_chunks:
            return []

        metadata_map = candidate_metadata_map or {}

        # Group chunks by video_id
        video_chunks_map: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in retrieved_chunks:
            vid = chunk.get("video_id")
            if not vid:
                continue
            if vid not in video_chunks_map:
                video_chunks_map[vid] = []
            video_chunks_map[vid].append(chunk)

        ranked_videos: List[Dict[str, Any]] = []

        for video_id, chunks in video_chunks_map.items():
            # Sort this video's chunks descending by similarity
            chunks.sort(key=lambda c: c.get("similarity_score", 0.0), reverse=True)

            best_chunk = chunks[0]
            max_score = float(best_chunk.get("similarity_score", 0.0))

            # Calculate bonus from secondary distinct matches (if any)
            if len(chunks) > 1:
                other_scores = [float(c.get("similarity_score", 0.0)) for c in chunks[1:4]]
                secondary_bonus = 0.05 * (sum(other_scores) / len(other_scores))
            else:
                secondary_bonus = 0.0

            aggregate_score = round(max_score + secondary_bonus, 4)

            # Retrieve video metadata
            meta = metadata_map.get(video_id, {})
            # If metadata was embedded inside the chunk, use that as fallback
            chunk_meta = best_chunk.get("metadata", {})

            video_result = {
                "video_id": video_id,
                "relevance_score": aggregate_score,
                "best_chunk_score": max_score,
                "title": meta.get("title") or chunk_meta.get("title", f"Video {video_id}"),
                "channel_name": meta.get("channel_name") or chunk_meta.get("channel_name", "YouTube"),
                "thumbnail_url": meta.get("thumbnail_url") or chunk_meta.get("thumbnail_url", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"),
                "published_at": meta.get("published_at") or chunk_meta.get("published_at", ""),
                "description": meta.get("description") or chunk_meta.get("description", ""),
                "best_chunk": best_chunk,
                "all_matching_chunks": chunks[:3]  # Keep top 3 supporting segments
            }
            ranked_videos.append(video_result)

        # Sort videos descending by relevance_score, then return only available top results.
        ranked_videos.sort(key=lambda v: v["relevance_score"], reverse=True)

        return ranked_videos[:self.top_k if top_k is None else top_k]
