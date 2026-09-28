"""In-process job queue so HTTP handlers never wait on inference (FR-API-003)."""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any, Callable

from backend.database.models import Job, Video, utc_now_iso
from backend.database.session import Database

log = logging.getLogger(__name__)

STAGE_PROGRESS = {
    "queued": 0.0,
    "ingest": 0.12,
    "detection": 0.32,
    "tracking": 0.55,
    "analytics": 0.78,
    "persist": 0.92,
    "completed": 1.0,
    "failed": 1.0,
}


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:8]}"


class JobBroker:
    """Queue job ids onto a daemon worker thread."""

    def __init__(self, handler: Callable[[str], None]) -> None:
        self._handler = handler
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="traffilytics-worker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                continue
            try:
                self._handler(item)
            except Exception:
                log.exception("Job %s failed in worker", item)


def create_job(
    db: Database,
    *,
    video_id: str,
    kind: str = "process",
    options: dict[str, Any] | None = None,
) -> Job:
    """Insert a queued job and return it (does not run the pipeline)."""
    job_id = new_job_id()
    with db.session_scope(write=True) as session:
        job = Job(
            job_id=job_id,
            video_id=video_id,
            kind=kind,
            status="queued",
            stage="queued",
            progress=0.0,
            options=options,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        session.add(job)
        session.flush()
        session.refresh(job)
        # Detach a snapshot so callers can read after the session closes.
        snapshot = Job(
            job_id=job.job_id,
            video_id=job.video_id,
            kind=job.kind,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            options=job.options,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
    return snapshot


def update_job(
    db: Database,
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    error: str | None = None,
    video_status: str | None = None,
) -> Job | None:
    with db.session_scope(write=True) as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        if status is not None:
            job.status = status
        if stage is not None:
            job.stage = stage
            if progress is None:
                job.progress = float(STAGE_PROGRESS.get(stage, job.progress))
        if progress is not None:
            job.progress = float(progress)
        if error is not None:
            job.error = error
        job.updated_at = utc_now_iso()
        if video_status is not None:
            video = session.get(Video, job.video_id)
            if video is not None:
                video.status = video_status
        session.flush()
        return Job(
            job_id=job.job_id,
            video_id=job.video_id,
            kind=job.kind,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            error=job.error,
            options=job.options,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


def get_job(db: Database, job_id: str) -> Job | None:
    with db.session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        return Job(
            job_id=job.job_id,
            video_id=job.video_id,
            kind=job.kind,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            error=job.error,
            options=job.options,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
