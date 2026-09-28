"""Model listing, evaluation ingest, and tracking diagnostics."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends

from backend.api.deps import get_db, get_settings
from backend.api.errors import ApiError
from backend.database.models import EvaluationRun, utc_now_iso
from backend.database.persist import get_video
from backend.database.session import Database
from backend.services.settings import Settings
from computer_vision.tracking.diagnostics import TrackingDiagnostics
from computer_vision.trajectories.generator import TrajectoryError, load_trajectories_json

router = APIRouter()


@router.get("/models")
def list_models(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    weights = settings.models_dir / "your_obb.pt"
    versions: list[dict[str, Any]] = []
    if weights.is_file():
        versions.append(
            {
                "model_version": "obb_v1",
                "weights": str(weights),
                "available": True,
            }
        )
    else:
        versions.append(
            {
                "model_version": "obb_v1",
                "weights": str(weights),
                "available": False,
                "note": "Train with scripts/train_obb.py; product inference needs models/your_obb.pt.",
            }
        )
    return {"models": versions}


@router.post("/models/train", status_code=501)
def train_model() -> dict[str, Any]:
    raise ApiError(
        501,
        "unsupported",
        "OBB training is a GPU CLI job: python scripts/train_obb.py --train-config models/configs/train_obb.yaml",
    )


@router.post("/models/{model_version}/evaluate")
def evaluate_model(
    model_version: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    metrics_path = settings.models_dir / "runs" / "eval_obb" / "metrics.json"
    if not metrics_path.is_file():
        raise ApiError(
            404,
            "not_found",
            f"No eval metrics at {metrics_path}. Run: python scripts/eval_obb.py --weights models/your_obb.pt",
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    eval_id = f"eval_{model_version}_{uuid.uuid4().hex[:8]}"
    with db.session_scope(write=True) as session:
        session.add(
            EvaluationRun(
                eval_id=eval_id,
                video_id=None,
                model_version=model_version,
                detection_metrics=payload if isinstance(payload, dict) else {"raw": payload},
                created_at=utc_now_iso(),
            )
        )
    detection = metrics if isinstance(metrics, dict) else {}
    return {
        "eval_id": eval_id,
        "model_version": model_version,
        "detection_metrics": {
            "mAP50": detection.get("mAP50"),
            "mAP50-95": detection.get("mAP50-95"),
            "precision": detection.get("precision")
            or (detection.get("raw") or {}).get("metrics/precision(B)"),
            "recall": detection.get("recall")
            or (detection.get("raw") or {}).get("metrics/recall(B)"),
            "per_class": detection.get("per_class") or {},
            "split": payload.get("split") if isinstance(payload, dict) else None,
        },
    }


@router.post("/videos/{video_id}/tracking-diagnostics")
def tracking_diagnostics(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        tracker_name = video.tracker_name
        model_version = video.model_version
    traj_path = settings.processed_dir / video_id / "trajectories.json"
    if not traj_path.is_file():
        raise ApiError(404, "not_found", f"No trajectories.json for {video_id}")
    try:
        trajectories, _meta = load_trajectories_json(traj_path)
    except TrajectoryError as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc
    report = TrackingDiagnostics().report(trajectories)
    eval_id = f"track_{video_id}_{uuid.uuid4().hex[:8]}"
    with db.session_scope(write=True) as session:
        session.add(
            EvaluationRun(
                eval_id=eval_id,
                video_id=video_id,
                model_version=model_version,
                tracker_name=tracker_name,
                tracking_diagnostics=report,
                created_at=utc_now_iso(),
            )
        )
    return {"eval_id": eval_id, "video_id": video_id, "tracking_diagnostics": report}


@router.get("/videos/{video_id}/evaluations")
def list_evaluations(video_id: str, db: Database = Depends(get_db)) -> dict[str, Any]:
    from sqlalchemy import select

    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        stmt = (
            select(EvaluationRun)
            .where(EvaluationRun.video_id == video_id)
            .order_by(EvaluationRun.created_at.desc())
        )
        rows = list(session.scalars(stmt))
        return {
            "video_id": video_id,
            "evaluations": [
                {
                    "eval_id": row.eval_id,
                    "model_version": row.model_version,
                    "tracker_name": row.tracker_name,
                    "detection_metrics": row.detection_metrics,
                    "tracking_diagnostics": row.tracking_diagnostics,
                    "created_at": row.created_at,
                }
                for row in rows
            ],
        }
