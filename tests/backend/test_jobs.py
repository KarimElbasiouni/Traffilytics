"""Job broker returns immediately; workers update status (FR-API-003)."""

from __future__ import annotations

import threading
import time

from backend.database.persist import upsert_video
from backend.services.jobs import JobBroker, create_job, get_job, update_job


def test_enqueue_does_not_block_on_handler(db) -> None:
    started = threading.Event()
    release = threading.Event()

    def handler(job_id: str) -> None:
        update_job(db, job_id, status="processing", stage="ingest", video_status="processing")
        started.set()
        release.wait(timeout=5)
        update_job(db, job_id, status="completed", stage="completed", progress=1.0, video_status="completed")

    with db.session_scope(write=True) as session:
        upsert_video(session, video_id="clip", status="queued", fps=10.0, resolution="64x48")

    broker = JobBroker(handler)
    broker.start()
    try:
        t0 = time.perf_counter()
        job = create_job(db, video_id="clip")
        broker.enqueue(job.job_id)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.25
        assert get_job(db, job.job_id).status == "queued"
        assert started.wait(timeout=2)
        inflight = get_job(db, job.job_id)
        assert inflight is not None
        assert inflight.status == "processing"
        assert inflight.stage == "ingest"
        release.set()
        deadline = time.time() + 2
        while time.time() < deadline:
            done = get_job(db, job.job_id)
            if done and done.status == "completed":
                break
            time.sleep(0.05)
        assert get_job(db, job.job_id).status == "completed"
    finally:
        release.set()
        broker.stop()
