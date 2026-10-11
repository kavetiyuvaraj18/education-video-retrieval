"""Central Retrieval Pipeline orchestrating YouTube retrieval, transcripts, embeddings, and FAISS ranking."""

import time
import logging
from typing import List, Dict, Any, Optional

from backend.config import MAX_CANDIDATE_VIDEOS, DEFAULT_TOP_K
from backend.youtube_client import YouTubeClient
from backend.transcript_extractor import TranscriptExtractor
from backend.preprocessor import TranscriptChunker
from backend.embedder import Embedder
from backend.vector_store import FAISSIndex
from backend.ranker import VideoRanker
from backend.timestamp_generator import TimestampGenerator
from backend.database import DatabaseManager

logger = logging.getLogger(__name__)


class VideoRetrievalPipeline:
    """End-to-end educational video retrieval and lecture portion localization pipeline."""

    def __init__(
        self,
        youtube_client: Optional[YouTubeClient] = None,
        transcript_extractor: Optional[TranscriptExtractor] = None,
        chunker: Optional[TranscriptChunker] = None,
        embedder: Optional[Embedder] = None,
        ranker: Optional[VideoRanker] = None,
        timestamp_generator: Optional[TimestampGenerator] = None,
        db: Optional[DatabaseManager] = None,
    ):
        self.youtube_client = youtube_client or YouTubeClient()
        self.transcript_extractor = transcript_extractor or TranscriptExtractor()
        self.chunker = chunker or TranscriptChunker()
        self.embedder = embedder or Embedder()
        self.ranker = ranker or VideoRanker(top_k=DEFAULT_TOP_K)
        self.timestamp_generator = timestamp_generator or TimestampGenerator()
        self.db = db or DatabaseManager()

    def process_query(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_limit: int = MAX_CANDIDATE_VIDEOS
    ) -> Dict[str, Any]:
        """
        Execute full pipeline:
        Query -> YouTube -> Transcripts -> Chunks -> Embeddings -> FAISS -> Ranking -> Timestamps

        Args:
            query: Student's natural language question.
            top_k: Number of ranked videos to return (default 6).
            candidate_limit: Max candidate videos to retrieve from YouTube.

        Returns:
            Dict containing query metadata and ranked results with timestamps.
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query}'")

        if not query or not query.strip():
            return {
                "query": query,
                "execution_time_seconds": 0.0,
                "total_candidates": 0,
                "total_chunks": 0,
                "total_results": 0,
                "results": []
            }

        # Step 1: YouTube candidate retrieval
        candidate_videos = self.youtube_client.search_videos(
            query=query,
            max_results=candidate_limit
        )
        logger.info(f"Retrieved {len(candidate_videos)} candidate videos from YouTube.")

        if not candidate_videos:
            return {
                "query": query,
                "execution_time_seconds": round(time.time() - start_time, 3),
                "total_candidates": 0,
                "total_chunks": 0,
                "total_results": 0,
                "results": []
            }

        candidate_metadata_map = {v["video_id"]: v for v in candidate_videos}
        for vid in candidate_videos:
            self.db.save_video_metadata(vid)

        # Step 2 & 3: Transcript Extraction and Chunking
        all_chunks: List[Dict[str, Any]] = []
        for vid_meta in candidate_videos:
            video_id = vid_meta["video_id"]
            try:
                segments = self.transcript_extractor.extract_transcript(
                    video_id=video_id,
                    allow_whisper_fallback=True
                )
                if not segments:
                    logger.info(f"No transcript available for video {video_id}; skipping.")
                    continue

                chunks = self.chunker.create_chunks(
                    video_id=video_id,
                    segments=segments,
                    video_metadata=vid_meta
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing transcript for video {video_id}: {e}", exc_info=True)
                continue

        logger.info(f"Generated {len(all_chunks)} overlapping transcript chunks in total.")

        if not all_chunks:
            logger.warning("No transcript chunks generated across all candidate videos.")
            return {
                "query": query,
                "execution_time_seconds": round(time.time() - start_time, 3),
                "total_candidates": len(candidate_videos),
                "total_chunks": 0,
                "total_results": 0,
                "results": []
            }

        # Step 4: Embedding generation
        chunk_texts = [c["text"] for c in all_chunks]
        chunk_embeddings = self.embedder.encode_texts(chunk_texts, normalize=True)
        query_embedding = self.embedder.encode_query(query, normalize=True)

        # Step 5: FAISS Semantic Retrieval
        dimension = self.embedder.get_dimension()
        faiss_store = FAISSIndex(dimension=dimension)
        faiss_store.add_chunks(chunk_embeddings, all_chunks)

        # Search top matching chunks (e.g. up to 30 to cover multiple videos)
        search_k = min(30, len(all_chunks))
        retrieved_chunks = faiss_store.search(query_embedding, top_k=search_k)

        # Step 6: Video Ranking & Deduplication
        ranked_videos = self.ranker.rank_videos(
            retrieved_chunks=retrieved_chunks,
            candidate_metadata_map=candidate_metadata_map
        )

        # Limit to top_k
        ranked_videos = ranked_videos[:top_k]

        # Step 7: Relevant Topic and Timestamp Generation
        enriched_results = self.timestamp_generator.enrich_all(ranked_videos)

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"Pipeline completed in {elapsed}s with {len(enriched_results)} ranked results.")

        # Persist query log to database
        self.db.log_query(query, len(enriched_results), elapsed)

        return {
            "query": query,
            "execution_time_seconds": elapsed,
            "total_candidates": len(candidate_videos),
            "total_chunks": len(all_chunks),
            "total_results": len(enriched_results),
            "results": enriched_results
        }
