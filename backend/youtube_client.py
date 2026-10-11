"""YouTube Data API v3 Client for Educational Video Retrieval."""

import logging
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from backend.config import YOUTUBE_API_KEY, MAX_CANDIDATE_VIDEOS

logger = logging.getLogger(__name__)

# Curated fallback educational videos with known public transcripts for offline/quota-exceeded testing
FALLBACK_EDUCATIONAL_VIDEOS: List[Dict[str, Any]] = [
    {
        "video_id": "aircAruvnKk",
        "title": "Neural Networks: But what is a neural network? | Chapter 1, Deep Learning",
        "description": "What are the neurons, why are there layers, and what is the math underlying it? 3Blue1Brown deep learning series.",
        "channel_name": "3Blue1Brown",
        "published_at": "2017-10-05T18:00:00Z",
        "thumbnail_url": "https://img.youtube.com/vi/aircAruvnKk/hqdefault.jpg"
    },
    {
        "video_id": "IHZwWFHWa-w",
        "title": "Gradient descent, how neural networks learn | Chapter 2, Deep Learning",
        "description": "What is gradient descent and backpropagation in machine learning and neural networks.",
        "channel_name": "3Blue1Brown",
        "published_at": "2017-10-15T18:00:00Z",
        "thumbnail_url": "https://img.youtube.com/vi/IHZwWFHWa-w/hqdefault.jpg"
    },
    {
        "video_id": "Ilg3gGewQ5U",
        "title": "What is backpropagation really doing? | Chapter 3, Deep Learning",
        "description": "The backpropagation algorithm visualized and explained with calculus and chain rule.",
        "channel_name": "3Blue1Brown",
        "published_at": "2017-11-03T18:00:00Z",
        "thumbnail_url": "https://img.youtube.com/vi/Ilg3gGewQ5U/hqdefault.jpg"
    },
    {
        "video_id": "tIeHLnjs5U8",
        "title": "Backpropagation calculus | Chapter 4, Deep Learning",
        "description": "The math behind backpropagation and partial derivatives in neural network training.",
        "channel_name": "3Blue1Brown",
        "published_at": "2017-11-03T18:00:00Z",
        "thumbnail_url": "https://img.youtube.com/vi/tIeHLnjs5U8/hqdefault.jpg"
    },
    {
        "video_id": "4b4MUYve_U8",
        "title": "Gradient Descent, Step-by-Step | StatQuest",
        "description": "Gradient Descent is a fundamental optimization algorithm in Machine Learning.",
        "channel_name": "StatQuest with Josh Starmer",
        "published_at": "2019-02-04T18:00:00Z",
        "thumbnail_url": "https://img.youtube.com/vi/4b4MUYve_U8/hqdefault.jpg"
    },
    {
        "video_id": "wjZofJX0v4U",
        "title": "Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!",
        "description": "Attention mechanism, positional encoding, and the Transformer architecture explained simply.",
        "channel_name": "StatQuest with Josh Starmer",
        "published_at": "2023-04-10T18:00:00Z",
        "thumbnail_url": "https://img.youtube.com/vi/wjZofJX0v4U/hqdefault.jpg"
    }
]


class YouTubeRetrievalError(Exception):
    """Custom exception raised for YouTube retrieval issues."""
    pass


class YouTubeClient:
    """Client for retrieving candidate educational videos via YouTube Data API v3."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else YOUTUBE_API_KEY
        self._service = None
        if self.api_key:
            try:
                self._service = build("youtube", "v3", developerKey=self.api_key)
            except Exception as e:
                logger.warning(f"Could not build YouTube API service with provided key: {e}")
                self._service = None

    def search_videos(
        self,
        query: str,
        max_results: int = MAX_CANDIDATE_VIDEOS,
        allow_fallback: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search YouTube for educational videos matching the student's query.

        Args:
            query: Natural language query.
            max_results: Maximum candidate videos to retrieve (up to 6).
            allow_fallback: If True, returns curated fallback videos when API is unavailable.

        Returns:
            List of video metadata dictionaries.
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided to YouTubeClient.")
            return []

        cleaned_query = query.strip()
        limit = max(1, min(max_results, MAX_CANDIDATE_VIDEOS))

        # Check if API client is initialized
        if not self._service or not self.api_key:
            logger.info("YouTube API key not configured. Using curated educational candidates.")
            if allow_fallback:
                return self._get_filtered_fallback(cleaned_query, limit)
            raise YouTubeRetrievalError("YouTube API key is missing or not configured.")

        try:
            # Append educational context hint if needed to prioritize high quality lectures
            search_request = self._service.search().list(
                q=f"{cleaned_query} lecture tutorial explanation",
                part="snippet",
                type="video",
                maxResults=limit,
                videoEmbeddable="true",
                safeSearch="moderate",
                relevanceLanguage="en"
            )
            response = search_request.execute()

            items = response.get("items", [])
            videos: List[Dict[str, Any]] = []

            for item in items:
                id_info = item.get("id", {})
                video_id = id_info.get("videoId")
                if not video_id:
                    continue

                snippet = item.get("snippet", {})
                thumbnails = snippet.get("thumbnails", {})
                # Pick highest available thumbnail
                thumb_url = (
                    thumbnails.get("high", {}).get("url")
                    or thumbnails.get("medium", {}).get("url")
                    or thumbnails.get("default", {}).get("url")
                    or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                )

                videos.append({
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_name": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail_url": thumb_url
                })

            if not videos and allow_fallback:
                logger.info("YouTube search returned 0 items; falling back to curated candidates.")
                return self._get_filtered_fallback(cleaned_query, limit)

            return videos[:limit]

        except HttpError as http_err:
            status_code = http_err.resp.status if hasattr(http_err, "resp") else None
            reason = getattr(http_err, "_get_reason", lambda: str(http_err))()
            logger.error(f"YouTube Data API HTTP error ({status_code}): {reason}")

            if allow_fallback:
                logger.info("Falling back to curated educational candidates due to API HTTP error.")
                return self._get_filtered_fallback(cleaned_query, limit)
            raise YouTubeRetrievalError(f"YouTube API error: {reason}") from http_err

        except Exception as e:
            logger.error(f"Unexpected error while querying YouTube API: {e}")
            if allow_fallback:
                return self._get_filtered_fallback(cleaned_query, limit)
            raise YouTubeRetrievalError(f"Failed to retrieve videos: {e}") from e

    def _get_filtered_fallback(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Filter curated fallback videos based on query terms or return top subset."""
        query_words = set(query.lower().split())
        scored_videos = []

        for vid in FALLBACK_EDUCATIONAL_VIDEOS:
            text = f"{vid['title']} {vid['description']}".lower()
            overlap = sum(1 for word in query_words if word in text)
            scored_videos.append((overlap, vid))

        scored_videos.sort(key=lambda x: x[0], reverse=True)
        return [vid for _, vid in scored_videos[:limit]]
