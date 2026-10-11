"""SQLite Database Module for Persistent Metadata and Query History."""

import sqlite3
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import SQLITE_DB_PATH, ENABLE_SQLITE_CACHE

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for query logs, video metadata, and cached transcripts."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self.enabled = ENABLE_SQLITE_CACHE
        if self.enabled:
            self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Video metadata cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS videos (
                        video_id TEXT PRIMARY KEY,
                        title TEXT,
                        channel_name TEXT,
                        description TEXT,
                        thumbnail_url TEXT,
                        published_at TEXT,
                        updated_at REAL
                    )
                """)
                # Transcript cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transcripts (
                        video_id TEXT PRIMARY KEY,
                        segments_json TEXT,
                        source TEXT,
                        created_at REAL
                    )
                """)
                # Search query history
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT,
                        results_count INTEGER,
                        execution_time REAL,
                        created_at REAL
                    )
                """)
                conn.commit()
                logger.info(f"Initialized SQLite database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")

    def save_video_metadata(self, video_dict: Dict[str, Any]):
        """Upsert video metadata."""
        if not self.enabled:
            return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO videos (
                        video_id, title, channel_name, description, thumbnail_url, published_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_dict.get("video_id"),
                    video_dict.get("title", ""),
                    video_dict.get("channel_name", ""),
                    video_dict.get("description", ""),
                    video_dict.get("thumbnail_url", ""),
                    video_dict.get("published_at", ""),
                    time.time()
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Error saving video metadata to DB: {e}")

    def save_transcript(self, video_id: str, segments: List[Dict[str, Any]], source: str = "caption"):
        """Save transcript segments JSON."""
        if not self.enabled:
            return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO transcripts (
                        video_id, segments_json, source, created_at
                    ) VALUES (?, ?, ?, ?)
                """, (
                    video_id,
                    json.dumps(segments, ensure_ascii=False),
                    source,
                    time.time()
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Error saving transcript to DB: {e}")

    def get_transcript(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached transcript segments if available."""
        if not self.enabled:
            return None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT segments_json FROM transcripts WHERE video_id = ?", (video_id,))
                row = cursor.fetchone()
                if row and row["segments_json"]:
                    return json.loads(row["segments_json"])
        except Exception as e:
            logger.warning(f"Error reading transcript from DB: {e}")
        return None

    def log_query(self, query: str, results_count: int, execution_time: float):
        """Log search query performance and metadata."""
        if not self.enabled:
            return
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_history (query, results_count, execution_time, created_at)
                    VALUES (?, ?, ?, ?)
                """, (query, results_count, execution_time, time.time()))
                conn.commit()
        except Exception as e:
            logger.warning(f"Error logging query to DB: {e}")
