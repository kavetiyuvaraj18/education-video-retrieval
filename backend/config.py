"""Configuration settings for Educational Video Retrieval System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# YouTube Data API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

# Hugging Face & Sentence Transformers
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# OpenAI Whisper configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")

# Server settings
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")

# Storage & Cache settings
CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", "data/cache")
SQLITE_DB_PATH = BASE_DIR / os.getenv("SQLITE_DB_PATH", "data/video_retrieval.db")
ENABLE_SQLITE_CACHE = os.getenv("ENABLE_SQLITE_CACHE", "true").lower() in ("true", "1", "yes")

# Retrieval & Chunking Hyperparameters
MAX_CANDIDATE_VIDEOS = 6
DEFAULT_TOP_K = 6
CHUNK_WORD_SIZE = 60
CHUNK_OVERLAP_WORDS = 15
SIMILARITY_METRIC = "cosine"  # Inner product with L2-normalized embeddings

# Ensure cache directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
