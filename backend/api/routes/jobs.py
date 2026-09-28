"""Job status polling (FR-API-003, FR-UI-008)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import get_db
from backend.api.errors import ApiError
from backend.database.session import Database
from backend.services.jobs import get_job

router = APIRouter()


@router.get("/jobs/{job_id}")
def job_status(job_id: str, db: Database = Depends(get_db)) -> dict[str, object]:
    job = get_job(db, job_id)
    if job is None:
        raise ApiError(404, "not_found", f"Unknown job_id: {job_id}")
    return {
        "job_id": job.job_id,
        "video_id": job.video_id,
        "stage": job.stage,
        "progress": job.progress,
        "status": job.status,
        "error": job.error,
        "kind": job.kind,
        "updated_at": job.updated_at,
    }
