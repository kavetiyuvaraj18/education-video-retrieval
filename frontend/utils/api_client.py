"""Frontend API client utility for interacting with FastAPI backend."""

import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BackendAPIClient:
    """Client for calling Educational Video Retrieval FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def check_health(self, timeout: float = 3.0) -> Dict[str, Any]:
        """Check backend health status."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            if resp.status_code == 200:
                return {"healthy": True, "data": resp.json()}
            return {"healthy": False, "error": f"Status code {resp.status_code}"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def search_videos(
        self,
        query: str,
        top_k: int = 6,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """
        Execute search query on FastAPI backend.

        Args:
            query: Question from the student.
            top_k: Number of ranked videos requested.
            timeout: Maximum wait time in seconds.

        Returns:
            Dict containing success flag, results list, and any error message.
        """
        try:
            payload = {"query": query, "top_k": top_k}
            resp = requests.post(f"{self.base_url}/search", json=payload, timeout=timeout)

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "results": data.get("results", []),
                    "execution_time": data.get("execution_time_seconds", 0.0),
                    "total_candidates": data.get("total_candidates", 0),
                    "total_chunks": data.get("total_chunks", 0),
                }

            error_msg = resp.text
            try:
                error_msg = resp.json().get("detail", error_msg)
            except Exception:
                pass

            return {"success": False, "error": f"Backend Error ({resp.status_code}): {error_msg}"}

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Search request timed out while processing video transcripts."}
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Could not connect to FastAPI backend. Ensure server is running at " + self.base_url
            }
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
