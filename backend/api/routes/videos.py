"""Videos, uploads, process triggers, lanes, detections, vehicles, trajectories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse

from backend.api.deps import get_broker, get_db, get_settings
from backend.api.errors import ApiError
from backend.api.schemas import LaneConfigBody, ProcessRequest
from backend.database.models import Video
from backend.database.persist import (
    get_trajectory_points,
    get_video,
    list_map_tracks,
    list_vehicles,
    list_videos,
    upsert_video,
)
from backend.database.session import Database
from backend.services.jobs import JobBroker, create_job
from backend.services.lanes import parse_lane_payload, write_lane_config
from backend.services.settings import Settings
from computer_vision.detection.detector import (
    DetectorError,
    frame_index_from_path,
    list_frame_images,
    load_detections_json,
)
from computer_vision.preprocessing.video_processor import VideoProcessor, VideoProcessorError
from computer_vision.trajectories.lanes import LaneConfigError

router = APIRouter()

_SAFE_ID = re.compile(r"[^A-Za-z0-9_\-]+")
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_FRAME_MEDIA = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _parse_resolution(value: str | None) -> tuple[int, int] | None:
    """Parse ``1920x1080`` into ``(width, height)``."""
    if not value or "x" not in value.lower():
        return None
    left, right = value.lower().split("x", 1)
    try:
        width, height = int(left.strip()), int(right.strip())
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _clip_frame_images(settings: Settings, video_id: str) -> list[Path]:
    """Return ingested stills for ``video_id``, or 404 if the folder is missing."""
    if "/" in video_id or "\\" in video_id or ".." in video_id:
        raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    frames_dir = (settings.processed_dir / video_id / "frames").resolve()
    root = settings.processed_dir.resolve()
    if root not in frames_dir.parents:
        raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    try:
        images = list_frame_images(frames_dir)
    except DetectorError as exc:
        raise ApiError(404, "not_found", f"No ingested frames for {video_id}") from exc
    if not images:
        raise ApiError(404, "not_found", f"No ingested frames for {video_id}")
    return images


def _select_frame(images: list[Path], offset: int | None) -> Path:
    """Pick a still by 0-based extracted-frame index; default is the middle still."""
    if offset is None:
        return images[len(images) // 2]
    if offset < 0 or offset >= len(images):
        raise ApiError(
            404,
            "not_found",
            f"Frame offset {offset} out of range (0–{len(images) - 1})",
        )
    return images[offset]


def _video_out(video: Video) -> dict[str, Any]:
    return {
        "video_id": video.video_id,
        "site": video.site,
        "location": video.location,
        "duration": video.duration,
        "fps": video.fps,
        "resolution": video.resolution,
        "source": video.source,
        "stabilized": video.stabilized,
        "status": video.status,
        "scale_m_per_px": video.scale_m_per_px,
        "lane_config": video.lane_config,
        "model_version": video.model_version,
        "tracker_name": video.tracker_name,
        "created_at": video.created_at,
    }


def _safe_video_id(filename: str) -> str:
    stem = Path(filename or "video").stem
    cleaned = _SAFE_ID.sub("_", stem).strip("_")
    return cleaned or "video"


def _unique_video_id(db: Database, base: str) -> str:
    with db.session_scope() as session:
        if session.get(Video, base) is None:
            return base
        for i in range(2, 10000):
            candidate = f"{base}_{i}"
            if session.get(Video, candidate) is None:
                return candidate
    return f"{base}_x"


def _peek_metadata(path: Path, video_id: str, site: str | None) -> dict[str, Any]:
    with VideoProcessor(path, video_id=video_id, site=site, source="upload") as processor:
        return processor.get_metadata()


@router.post("/videos", status_code=202)
async def upload_video(
    file: UploadFile = File(...),
    site: str | None = Form(default=None),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
    broker: JobBroker = Depends(get_broker),
) -> dict[str, Any]:
    filename = file.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in _VIDEO_EXTS:
        raise ApiError(400, "invalid_request", f"Unsupported video type: {ext or '(none)'}")

    video_id = _unique_video_id(db, _safe_video_id(filename))
    dest = settings.raw_dir / f"{video_id}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = await file.read()
    if not payload:
        raise ApiError(400, "invalid_request", "Uploaded file is empty")
    dest.write_bytes(payload)

    try:
        meta = _peek_metadata(dest, video_id, site)
    except VideoProcessorError as exc:
        dest.unlink(missing_ok=True)
        raise ApiError(400, "invalid_request", str(exc)) from exc

    with db.session_scope(write=True) as session:
        upsert_video(
            session,
            video_id=video_id,
            site=site if site is not None else meta.get("site"),
            duration=meta.get("duration"),
            fps=meta.get("fps"),
            resolution=meta.get("resolution"),
            source="upload",
            stabilized=bool(meta.get("stabilized")),
            status="queued",
        )

    job = create_job(db, video_id=video_id, kind="process")
    broker.enqueue(job.job_id)
    return {
        "video_id": video_id,
        "job_id": job.job_id,
        "site": site if site is not None else meta.get("site"),
        "fps": meta.get("fps"),
        "resolution": meta.get("resolution"),
        "duration": meta.get("duration"),
        "source": "upload",
        "status": "queued",
    }


@router.get("/videos")
def videos_index(db: Database = Depends(get_db)) -> dict[str, Any]:
    with db.session_scope() as session:
        rows = list_videos(session)
        return {"videos": [_video_out(v) for v in rows]}


@router.get("/videos/{video_id}")
def video_detail(video_id: str, db: Database = Depends(get_db)) -> dict[str, Any]:
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        return _video_out(video)


@router.post("/videos/{video_id}/process", status_code=202)
def process_video(
    video_id: str,
    body: ProcessRequest | None = None,
    db: Database = Depends(get_db),
    broker: JobBroker = Depends(get_broker),
) -> dict[str, Any]:
    req = body or ProcessRequest()
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        if video.status == "processing":
            raise ApiError(409, "conflict", f"{video_id} is already processing")
    options = req.model_dump()
    job = create_job(db, video_id=video_id, kind="process", options=options)
    with db.session_scope(write=True) as session:
        row = session.get(Video, video_id)
        if row is not None:
            row.status = "queued"
    broker.enqueue(job.job_id)
    return {
        "video_id": video_id,
        "job_id": job.job_id,
        "status": "queued",
        "options": options,
    }


@router.put("/videos/{video_id}/lanes")
def put_lanes(
    video_id: str,
    body: LaneConfigBody,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    try:
        assigner = parse_lane_payload(body.model_dump(), video_id=video_id)
    except LaneConfigError as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc
    path = write_lane_config(assigner, video_id=video_id, lanes_dir=settings.lanes_dir)
    payload = assigner.to_dict()
    with db.session_scope(write=True) as session:
        row = session.get(Video, video_id)
        if row is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        row.lane_config = payload
    return {"video_id": video_id, "lane_config": payload, "path": str(path)}


@router.get("/videos/{video_id}/detections")
def get_detections(
    video_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    dest = settings.processed_dir / video_id / "detections.json"
    if not dest.is_file():
        raise ApiError(404, "not_found", f"No detections.json for {video_id}")
    try:
        detections, meta = load_detections_json(dest)
    except DetectorError as exc:
        raise ApiError(400, "invalid_request", str(exc)) from exc
    rows = [d.to_dict() for d in detections]
    sliced = rows[offset : offset + limit]
    return {
        "video_id": video_id,
        "offset": offset,
        "limit": limit,
        "n_detections": len(rows),
        "detections": sliced,
        "meta": {k: v for k, v in meta.items() if k != "detections"},
    }


@router.get("/videos/{video_id}/vehicles")
def get_vehicles(video_id: str, db: Database = Depends(get_db)) -> dict[str, Any]:
    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        vehicles = list_vehicles(session, video_id)
        return {
            "video_id": video_id,
            "n_vehicles": len(vehicles),
            "vehicles": [
                {
                    "video_id": v.video_id,
                    "track_id": v.track_id,
                    "class_id": v.class_id,
                    "vehicle_type": v.vehicle_type,
                    "entry_frame": v.entry_frame,
                    "exit_frame": v.exit_frame,
                    "entry_time": v.entry_time,
                    "exit_time": v.exit_time,
                }
                for v in vehicles
            ],
        }


@router.get("/videos/{video_id}/preview")
def get_preview(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """JSON for the intersection still: size, extracted-frame count, default index."""
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        size = _parse_resolution(video.resolution)
    images = _clip_frame_images(settings, video_id)
    default_i = len(images) // 2
    chosen = images[default_i]
    if size is None:
        import cv2

        arr = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if arr is not None:
            size = (int(arr.shape[1]), int(arr.shape[0]))
    width, height = size if size is not None else (None, None)
    indices = [frame_index_from_path(path, fallback=i) for i, path in enumerate(images)]
    return {
        "video_id": video_id,
        "n_frames": len(images),
        "default_i": default_i,
        "frame": frame_index_from_path(chosen, fallback=default_i),
        "indices": indices,
        "width": width,
        "height": height,
        "overlay": (settings.processed_dir / video_id / "tracks_overlay.mp4").is_file(),
    }


@router.get("/videos/{video_id}/frame")
def get_frame(
    video_id: str,
    i: int | None = Query(default=None, ge=0),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve one ingested JPEG/PNG so the dashboard can draw on the aerial still."""
    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    path = _select_frame(_clip_frame_images(settings, video_id), i)
    media = _FRAME_MEDIA.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/videos/{video_id}/overlay")
def get_overlay(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve a boxed overlay MP4 (OBB + track_id on each ingested frame)."""
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        fps = float(video.fps or 10.0)
    if "/" in video_id or "\\" in video_id or ".." in video_id:
        raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    processed = (settings.processed_dir / video_id).resolve()
    root = settings.processed_dir.resolve()
    if root not in processed.parents and processed != root:
        raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    dest = processed / "tracks_overlay.mp4"
    traj_path = processed / "trajectories.json"
    frames_dir = processed / "frames"
    det_path = processed / "detections.json"
    stale = (not dest.is_file()) or dest.stat().st_size <= 0
    if dest.is_file():
        dest_mtime = dest.stat().st_mtime
        for src in (traj_path, det_path):
            if src.is_file() and src.stat().st_mtime > dest_mtime:
                stale = True
                break
    if stale:
        from computer_vision.tracking.overlay import ensure_overlay_video, overlay_playback_fps
        from computer_vision.trajectories.generator import (
            TrajectoryError,
            load_trajectories_json,
        )

        traj_path = processed / "trajectories.json"
        frames_dir = processed / "frames"
        if not traj_path.is_file() or not frames_dir.is_dir():
            raise ApiError(404, "not_found", f"No overlay video for {video_id}")
        try:
            trajectories, _meta = load_trajectories_json(traj_path)
            detections = []
            det_path = processed / "detections.json"
            if det_path.is_file():
                detections, _det_meta = load_detections_json(det_path)
            track_cfg = settings.config.get("tracking") or {}
            n_frames = len(list_frame_images(frames_dir))
            duration = None
            meta_path = processed / "metadata.json"
            if meta_path.is_file():
                try:
                    duration = float(json.loads(meta_path.read_text(encoding="utf-8")).get("duration") or 0) or None
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    duration = None
            overlay_fps = overlay_playback_fps(
                n_frames,
                duration,
                float(track_cfg.get("overlay_fps") or fps or 10),
            )
            dest = ensure_overlay_video(
                frames_dir,
                trajectories,
                dest,
                detections=detections,
                fps=overlay_fps,
                force=True,
            )
        except (DetectorError, TrajectoryError, OSError) as exc:
            raise ApiError(404, "not_found", f"Could not build overlay for {video_id}: {exc}") from exc
    return FileResponse(
        dest,
        media_type="video/mp4",
        filename=dest.name,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/videos/{video_id}/map")
def get_map(
    video_id: str,
    stride: int = Query(4, ge=1, le=50),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        return list_map_tracks(session, video_id, stride=stride)


@router.get("/videos/{video_id}/vehicles/{track_id}/trajectory")
def get_trajectory(
    video_id: str,
    track_id: int,
    stride: int = Query(1, ge=1, le=100),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        points = get_trajectory_points(session, video_id, track_id, stride=stride)
        if not points:
            raise ApiError(404, "not_found", f"No trajectory for track_id={track_id}")
        return {
            "track_id": track_id,
            "class_id": points[0].class_id,
            "stride": stride,
            "points": [
                {
                    "frame": p.frame,
                    "center_x": p.center_x,
                    "center_y": p.center_y,
                    "lane": p.lane,
                    "angle": p.angle,
                    "confidence": p.confidence,
                }
                for p in points
            ],
        }


def load_analytics_json(db: Database, settings: Settings, video_id: str) -> dict[str, Any]:
    """Prefer the persisted report blob; fall back to processed/analytics.json."""
    from backend.database.models import Report

    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        report = session.get(Report, video_id)
        if report is not None and report.analytics_json:
            return dict(report.analytics_json)
    path = settings.processed_dir / video_id / "analytics.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    raise ApiError(404, "not_found", f"No analytics for {video_id}")
