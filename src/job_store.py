"""
Job Store - SQLite-backed job history and log management
"""

import os
import json
import sqlite3
import logging
import threading
from datetime import datetime
from typing import Optional, List
from queue import Queue, Empty

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("JOB_DB_PATH", "data/jobs.db")


class JobStore:
    """SQLite-backed job storage with live log streaming."""

    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()
        # Per-job log queues for SSE streaming
        self._log_queues: dict[str, list[Queue]] = {}
        self._lock = threading.Lock()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'pending',
                    step TEXT DEFAULT '',
                    step_number INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 5,
                    title TEXT DEFAULT '',
                    content_type TEXT DEFAULT '',
                    language TEXT DEFAULT '',
                    youtube_url TEXT DEFAULT '',
                    youtube_id TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    created_at TEXT DEFAULT '',
                    completed_at TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    level TEXT,
                    message TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
            """)
            conn.commit()

    def create_job(self, job_id: str, content_type: str = "", language: str = "") -> dict:
        """Create a new job record."""
        now = datetime.now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, content_type, language, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, "running", content_type, language, now)
            )
            conn.commit()
        return {"job_id": job_id, "status": "running", "created_at": now}

    def update_job(self, job_id: str, **kwargs):
        """Update job fields."""
        valid_fields = {"status", "step", "step_number", "title", "youtube_url", "youtube_id", "error", "completed_at"}
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [job_id]
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
            conn.commit()

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get a single job."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_jobs(self, limit: int = 50) -> List[dict]:
        """Get recent jobs."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_running_job(self) -> Optional[dict]:
        """Get the currently running job, if any."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'running' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def add_log(self, job_id: str, level: str, message: str):
        """Add a log entry and broadcast to SSE listeners."""
        now = datetime.now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO job_logs (job_id, level, message, timestamp) VALUES (?, ?, ?, ?)",
                (job_id, level, message, now)
            )
            conn.commit()

        # Broadcast to SSE listeners
        log_entry = {"level": level, "message": message, "timestamp": now}
        with self._lock:
            if job_id in self._log_queues:
                for q in self._log_queues[job_id]:
                    q.put(log_entry)

    def get_logs(self, job_id: str, limit: int = 500) -> List[dict]:
        """Get logs for a job."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM job_logs WHERE job_id = ? ORDER BY id ASC LIMIT ?",
                (job_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def subscribe(self, job_id: str) -> Queue:
        """Subscribe to live logs for a job."""
        q = Queue()
        with self._lock:
            if job_id not in self._log_queues:
                self._log_queues[job_id] = []
            self._log_queues[job_id].append(q)
        return q

    def unsubscribe(self, job_id: str, q: Queue):
        """Unsubscribe from live logs."""
        with self._lock:
            if job_id in self._log_queues:
                self._log_queues[job_id] = [
                    x for x in self._log_queues[job_id] if x is not q
                ]
                if not self._log_queues[job_id]:
                    del self._log_queues[job_id]


class JobLogHandler(logging.Handler):
    """Custom log handler that captures logs into the JobStore."""

    def __init__(self, store: JobStore, job_id: str):
        super().__init__()
        self.store = store
        self.job_id = job_id

    def emit(self, record):
        try:
            msg = self.format(record)
            self.store.add_log(self.job_id, record.levelname, msg)
        except Exception:
            pass


# Global singleton
_store: Optional[JobStore] = None

def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
