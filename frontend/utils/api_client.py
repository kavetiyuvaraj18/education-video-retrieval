
"""Frontend API client for the FastAPI backend."""

import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BackendAPIClient:
    """Client for the Educational Video Retrieval backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def check_health(self, timeout: float = 15.0) -> Dict[str, Any]:
        """Check backend health with a longer timeout."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=timeout,
            )

            if response.status_code == 200:
                return {
                    "healthy": True,
                    "data": response.json(),
                }

            return {
                "healthy": False,
                "error": f"Status code {response.status_code}",
            }

        except requests.exceptions.Timeout:
            return {
                "healthy": False,
                "error": (
                    f"Backend health check timed out after {timeout} seconds. "
                    "The backend may be busy."
                ),
            }

        except requests.exceptions.ConnectionError:
            return {
                "healthy": False,
                "error": (
                    f"Cannot connect to backend at {self.base_url}. "
                    "Check that the backend is running."
                ),
            }

        except Exception as exc:
            logger.exception("Backend health check failed")
            return {
                "healthy": False,
                "error": str(exc),
            }

    def search_videos(
        self,
        query: str,
        top_k: int = 6,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Search for relevant educational videos and timestamps."""
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "candidate_limit": max(top_k + 2, 8),  # always fetch extra candidates
                },
                timeout=timeout,
            )

            if response.status_code == 200:
                data = response.json()

                return {
                    "success": True,
                    "results": data.get("results", []),
                    "execution_time": data.get(
                        "execution_time_seconds", 0.0
                    ),
                    "total_candidates": data.get(
                        "total_candidates", 0
                    ),
                    "total_chunks": data.get("total_chunks", 0),
                }

            try:
                error_msg = response.json().get(
                    "detail", response.text
                )
            except ValueError:
                error_msg = response.text

            return {
                "success": False,
                "error": (
                    f"Backend Error ({response.status_code}): "
                    f"{error_msg}"
                ),
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": (
                    "Search timed out while processing video transcripts. "
                    "Please try again."
                ),
            }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": (
                    f"Could not connect to the backend at {self.base_url}. "
                    "Ensure the server is running."
                ),
            }

        except Exception as exc:
            logger.exception("Video search failed")
            return {
                "success": False,
                "error": f"Unexpected error: {exc}",
            }
