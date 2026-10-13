"""Central Retrieval Pipeline orchestrating YouTube retrieval, transcripts, embeddings, and FAISS ranking."""

import hashlib
import json
import logging
import os
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import List, Dict, Any, Optional

import numpy as np

from backend.config import CACHE_DIR, MAX_CANDIDATE_VIDEOS, DEFAULT_TOP_K
from backend.youtube_client import YouTubeClient
from backend.transcript_extractor import TranscriptExtractor
from backend.preprocessor import TranscriptChunker
from backend.embedder import Embedder
from backend.vector_store import FAISSIndex
from backend.ranker import VideoRanker
from backend.timestamp_generator import TimestampGenerator
from backend.database import DatabaseManager

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_WORKERS = 4
MAX_WHISPER_FALLBACKS_PER_SEARCH = 2
MAX_CACHED_FAISS_INDEXES = 2
CHUNK_CACHE_VERSION = 1
INDEX_CACHE_VERSION = 1


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
        self._processed_cache_dir = CACHE_DIR / "processed"
        self._chunk_cache_dir = self._processed_cache_dir / "chunks"
        self._embedding_cache_dir = self._processed_cache_dir / "embeddings"
        self._chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_cache_lock = RLock()
        self._faiss_indexes: OrderedDict[str, FAISSIndex] = OrderedDict()
        self._faiss_cache_lock = RLock()

    @staticmethod
    def _digest(value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _chunk_cache_key(self, video_id: str, segments: List[Dict[str, Any]]) -> str:
        return self._digest({
            "version": CHUNK_CACHE_VERSION,
            "video_id": video_id,
            "segments": segments,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
        })

    def _get_video_chunks(
        self,
        video_metadata: Dict[str, Any],
        segments: List[Dict[str, Any]],
        chunk_key: str,
    ) -> tuple[List[Dict[str, Any]], bool]:
        video_id = video_metadata["video_id"]
        cache_path = self._chunk_cache_dir / f"{chunk_key}.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("key") == chunk_key and isinstance(cached.get("chunks"), list):
                chunks = cached["chunks"]
                for chunk in chunks:
                    chunk["metadata"] = video_metadata
                return chunks, True
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Ignoring invalid chunk cache for %s: %s", video_id, exc)

        chunks = self.chunker.create_chunks(
            video_id=video_id,
            segments=segments,
            video_metadata=video_metadata,
        )
        self._write_json_cache(cache_path, {"key": chunk_key, "chunks": chunks})
        return chunks, False

    @staticmethod
    def _write_json_cache(path: Path, value: Dict[str, Any]) -> None:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
                suffix=".tmp",
            ) as cache:
                temp_path = Path(cache.name)
                json.dump(value, cache, ensure_ascii=False)
            os.replace(temp_path, path)
        except OSError as exc:
            logger.warning("Could not persist retrieval cache %s: %s", path.name, exc)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _embedding_cache_key(
        self,
        video_id: str,
        chunks: List[Dict[str, Any]],
        model_identity: str,
    ) -> str:
        return self._digest({
            "version": INDEX_CACHE_VERSION,
            "video_id": video_id,
            "model": model_identity,
            "chunks": [
                (
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["start_time"],
                    chunk["end_time"],
                )
                for chunk in chunks
            ],
        })

    def _read_embedding_cache(
        self,
        cache_key: str,
        expected_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        cache_path = self._embedding_cache_dir / f"{cache_key}.npy"
        try:
            vectors = np.load(cache_path, allow_pickle=False)
            if vectors.shape != expected_shape or not np.isfinite(vectors).all():
                logger.warning("Ignoring stale or invalid embedding cache %s.", cache_path.name)
                return None
            return np.asarray(vectors, dtype=np.float32)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Ignoring unreadable embedding cache %s: %s", cache_path.name, exc)
            return None

    def _write_embedding_cache(self, cache_key: str, vectors: np.ndarray) -> None:
        cache_path = self._embedding_cache_dir / f"{cache_key}.npy"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=self._embedding_cache_dir,
                delete=False,
                suffix=".npy",
            ) as cache:
                temp_path = Path(cache.name)
                np.save(cache, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
            os.replace(temp_path, cache_path)
        except OSError as exc:
            logger.warning("Could not persist embedding cache %s: %s", cache_path.name, exc)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _get_faiss_index(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        model_identity: str,
    ) -> tuple[FAISSIndex, bool]:
        corpus_key = self._digest({
            "version": INDEX_CACHE_VERSION,
            "model": model_identity,
            "chunks": [
                (
                    chunk["video_id"],
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["start_time"],
                    chunk["end_time"],
                )
                for chunk in chunks
            ],
        })
        with self._faiss_cache_lock:
            cached_index = self._faiss_indexes.get(corpus_key)
            if cached_index is not None:
                self._faiss_indexes.move_to_end(corpus_key)
                return cached_index, True

            index = FAISSIndex(dimension=embeddings.shape[1])
            index.add_chunks(embeddings, chunks)
            self._faiss_indexes[corpus_key] = index
            while len(self._faiss_indexes) > MAX_CACHED_FAISS_INDEXES:
                self._faiss_indexes.popitem(last=False)
            return index, False

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
        start_time = time.perf_counter()
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
        youtube_start = time.perf_counter()
        candidate_videos = self.youtube_client.search_videos(
            query=query,
            max_results=candidate_limit
        )
        youtube_seconds = time.perf_counter() - youtube_start
        unique_candidates = {}
        for video in candidate_videos:
            unique_candidates.setdefault(video["video_id"], video)
        candidate_videos = list(unique_candidates.values())
        logger.info(f"Retrieved {len(candidate_videos)} candidate videos from YouTube.")

        if not candidate_videos:
            return {
                "query": query,
                "execution_time_seconds": round(time.perf_counter() - start_time, 3),
                "total_candidates": 0,
                "total_chunks": 0,
                "total_results": 0,
                "results": []
            }

        candidate_metadata_map = {v["video_id"]: v for v in candidate_videos}
        for vid in candidate_videos:
            self.db.save_video_metadata(vid)

        def load_video_chunks(
            video_metadata: Dict[str, Any],
            allow_whisper_fallback: bool,
        ):
            video_id = video_metadata["video_id"]
            transcript_start = time.perf_counter()
            try:
                segments = self.transcript_extractor.extract_transcript(
                    video_id=video_id,
                    allow_whisper_fallback=allow_whisper_fallback,
                )
                transcript_seconds = time.perf_counter() - transcript_start
                if not segments:
                    logger.info(f"No transcript available for video {video_id}; skipping.")
                    return video_id, [], transcript_seconds, 0.0, False
                chunk_start = time.perf_counter()
                cache_key = self._chunk_cache_key(video_id, segments)
                chunks, cache_hit = self._get_video_chunks(
                    video_metadata,
                    segments,
                    cache_key,
                )
                chunk_seconds = time.perf_counter() - chunk_start
                return video_id, chunks, transcript_seconds, chunk_seconds, cache_hit
            except Exception as e:
                logger.error(f"Error processing transcript for video {video_id}: {e}", exc_info=True)
                return video_id, [], time.perf_counter() - transcript_start, 0.0, False

        # Network transcript requests can progress independently; constrain their
        # concurrency and retain input order for deterministic tie handling.
        transcript_wall_start = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=min(MAX_TRANSCRIPT_WORKERS, len(candidate_videos)),
            thread_name_prefix="transcript",
        ) as executor:
            video_outputs = list(executor.map(
                load_video_chunks,
                candidate_videos,
                (
                    index < MAX_WHISPER_FALLBACKS_PER_SEARCH
                    for index in range(len(candidate_videos))
                ),
            ))
        transcript_wall_seconds = time.perf_counter() - transcript_wall_start
        transcript_work_seconds = sum(result[2] for result in video_outputs)
        chunk_work_seconds = sum(result[3] for result in video_outputs)
        chunk_cache_hits = sum(result[4] for result in video_outputs)
        all_chunks: List[Dict[str, Any]] = [
            chunk for _, chunks, _, _, _ in video_outputs for chunk in chunks
        ]
        logger.info(f"Generated {len(all_chunks)} overlapping transcript chunks in total.")
        logger.info(
            "Transcript stages: wall=%.3fs worker_time=%.3fs chunk_work=%.3fs "
            "chunk_cache_hits=%d/%d",
            transcript_wall_seconds,
            transcript_work_seconds,
            chunk_work_seconds,
            chunk_cache_hits,
            len(candidate_videos),
        )
        logger.info(
            "Whisper fallback enabled for at most %d candidate videos in this search.",
            min(MAX_WHISPER_FALLBACKS_PER_SEARCH, len(candidate_videos)),
        )

        if not all_chunks:
            logger.warning("No transcript chunks generated across all candidate videos.")
            return {
                "query": query,
                "execution_time_seconds": round(time.perf_counter() - start_time, 3),
                "total_candidates": len(candidate_videos),
                "total_chunks": 0,
                "total_results": 0,
                "results": []
            }

        # Step 4: Embedding generation
        embedding_start = time.perf_counter()
        dimension = self.embedder.get_dimension()
        model_identity = getattr(
            self.embedder,
            "cache_identity",
            f"{getattr(self.embedder, 'model_name', type(self.embedder).__qualname__)}|{dimension}",
        )
        video_chunks: List[tuple[str, List[Dict[str, Any]]]] = []
        for video_id, chunks, _, _, _ in video_outputs:
            if chunks:
                video_chunks.append((video_id, chunks))

        # Keep cache lookup, generation, and persistence atomic per pipeline so
        # overlapping searches cannot encode the same cold chunks concurrently.
        with self._embedding_cache_lock:
            chunk_embeddings = np.empty((len(all_chunks), dimension), dtype=np.float32)
            missing_indices: List[int] = []
            embedding_cache_hits = 0
            cache_entries = []
            offset = 0
            for video_id, chunks in video_chunks:
                cache_key = self._embedding_cache_key(video_id, chunks, model_identity)
                cached_vectors = self._read_embedding_cache(cache_key, (len(chunks), dimension))
                cache_entries.append((cache_key, offset, len(chunks), cached_vectors is None))
                if cached_vectors is not None:
                    chunk_embeddings[offset:offset + len(chunks)] = cached_vectors
                    embedding_cache_hits += 1
                else:
                    missing_indices.extend(range(offset, offset + len(chunks)))
                offset += len(chunks)

            if missing_indices:
                encoded_vectors = self.embedder.encode_texts(
                    [all_chunks[index]["text"] for index in missing_indices],
                    normalize=True,
                )
                if encoded_vectors.shape != (len(missing_indices), dimension):
                    raise ValueError(
                        "Embedder returned an unexpected shape: "
                        f"{encoded_vectors.shape}, expected {(len(missing_indices), dimension)}."
                    )
                chunk_embeddings[missing_indices] = encoded_vectors
                for cache_key, offset, count, missing in cache_entries:
                    if missing:
                        self._write_embedding_cache(
                            cache_key,
                            chunk_embeddings[offset:offset + count],
                        )
        query_embedding = self.embedder.encode_query(query, normalize=True)
        embedding_seconds = time.perf_counter() - embedding_start

        # Step 5: FAISS Semantic Retrieval
        index_start = time.perf_counter()
        faiss_store, index_cache_hit = self._get_faiss_index(
            chunk_embeddings,
            all_chunks,
            model_identity,
        )
        index_build_seconds = time.perf_counter() - index_start

        # Expand retrieval when repeated chunks from a few videos hide other
        # distinct candidates from the initial set.
        search_k = min(30, len(all_chunks))
        faiss_search_seconds = 0.0
        ranking_seconds = 0.0
        while True:
            search_start = time.perf_counter()
            retrieved_chunks = faiss_store.search(query_embedding, top_k=search_k)
            faiss_search_seconds += time.perf_counter() - search_start
            ranking_start = time.perf_counter()
            ranked_videos = self.ranker.rank_videos(
                retrieved_chunks=retrieved_chunks,
                candidate_metadata_map=candidate_metadata_map,
                top_k=top_k,
            )
            ranking_seconds += time.perf_counter() - ranking_start
            if len(ranked_videos) >= top_k or search_k >= len(all_chunks):
                break
            search_k = min(search_k * 2, len(all_chunks))

        # Step 7: Relevant Topic and Timestamp Generation
        timestamp_start = time.perf_counter()
        enriched_results = self.timestamp_generator.enrich_all(ranked_videos)
        timestamp_seconds = time.perf_counter() - timestamp_start

        elapsed = round(time.perf_counter() - start_time, 3)
        logger.info(f"Pipeline completed in {elapsed}s with {len(enriched_results)} ranked results.")
        logger.info(
            "Search stage timings: youtube=%.3fs transcript_wall=%.3fs "
            "embedding=%.3fs embedding_cache_hits=%d/%d index_build=%.3fs "
            "faiss_search=%.3fs ranking=%.3fs timestamps_response=%.3fs "
            "faiss_index_cache_hit=%s total=%.3fs",
            youtube_seconds,
            transcript_wall_seconds,
            embedding_seconds,
            embedding_cache_hits,
            len(video_chunks),
            index_build_seconds,
            faiss_search_seconds,
            ranking_seconds,
            timestamp_seconds,
            index_cache_hit,
            elapsed,
        )

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
