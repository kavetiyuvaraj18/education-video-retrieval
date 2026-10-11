"""FastAPI REST API for Educational Video Retrieval."""

import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import (
    EMBEDDING_MODEL_NAME,
    MAX_CANDIDATE_VIDEOS,
    DEFAULT_TOP_K,
)
from backend.pipeline import VideoRetrievalPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Educational Video Retrieval API",
    description="API for retrieving relevant educational videos and pinpointing exact lecture timestamps.",
    version="1.0.0"
)

# CORS middleware for Streamlit / external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy pipeline singleton
_pipeline: Optional[VideoRetrievalPipeline] = None


def get_pipeline() -> VideoRetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing VideoRetrievalPipeline...")
        _pipeline = VideoRetrievalPipeline()
    return _pipeline


# Pydantic Schemas
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Student conceptual question")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10, description="Number of videos to return")
    candidate_limit: int = Field(default=MAX_CANDIDATE_VIDEOS, ge=1, le=12, description="Candidate search limit")


class VideoResultItem(BaseModel):
    video_id: str
    title: str
    channel_name: str
    thumbnail_url: str
    published_at: Optional[str] = ""
    description: Optional[str] = ""
    relevance_score: float
    start_time_seconds: float
    end_time_seconds: float
    start_time_formatted: str
    end_time_formatted: str
    topic_text: str
    video_url: str
    timestamp_url: str


class SearchResponse(BaseModel):
    query: str
    execution_time_seconds: float
    total_candidates: int
    total_chunks: int
    total_results: int
    results: List[VideoResultItem]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    service: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "service": "Educational Video Retrieval Backend"
    }


@app.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """
    Search educational videos and retrieve timestamped lecture portions.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter cannot be empty."
        )

    try:
        pipeline = get_pipeline()
        pipeline_output = pipeline.process_query(
            query=query,
            top_k=request.top_k,
            candidate_limit=request.candidate_limit
        )

        return pipeline_output

    except Exception as e:
        logger.error(f"Error during video search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the search: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    from backend.config import BACKEND_HOST, BACKEND_PORT
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
