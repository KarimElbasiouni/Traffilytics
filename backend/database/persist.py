"""Write generated trajectories, analytics, and events into SQL tables."""

from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from analytics.types import clock_time
from backend.database.models import (
    AnalyticsRow,
    Event,
    EvaluationRun,
    Report,
    TrajectoryRow,
    Vehicle,
    Video,
    utc_now_iso,
)
from backend.services.reports import build_report
from computer_vision.detection.types import CLASS_NAMES
from computer_vision.trajectories.types import Trajectory

_STATE_SCORE = {
    "queued": 1.0,
    "congested": 0.7,
    "free": 0.15,
    "unknown": None,
}


class PersistError(Exception):
    """Raised when a video row is missing or analytics payload is unusable."""


def get_video(session: Session, video_id: str) -> Video | None:
    return session.get(Video, video_id)


def require_video(session: Session, video_id: str) -> Video:
    video = get_video(session, video_id)
    if video is None:
        raise PersistError(f"Unknown video_id: {video_id}")
    return video


def upsert_video(
    session: Session,
    *,
    video_id: str,
    site: str | None = None,
    location: str | None = None,
    duration: float | None = None,
    fps: float | None = None,
    resolution: str | None = None,
    source: str | None = None,
    stabilized: bool | None = None,
    status: str | None = None,
    scale_m_per_px: float | None = None,
    lane_config: dict[str, Any] | None = None,
    model_version: str | None = None,
    tracker_name: str | None = None,
) -> Video:
    """Insert or update a videos row (FR-DB-001)."""
    video = session.get(Video, video_id)
    if video is None:
        video = Video(video_id=video_id, created_at=utc_now_iso())
        session.add(video)
        session.flush()
    if site is not None:
        video.site = site
    if location is not None:
        video.location = location
    if duration is not None:
        video.duration = duration
    if fps is not None:
        video.fps = fps
    if resolution is not None:
        video.resolution = resolution
    if source is not None:
        video.source = source
    if stabilized is not None:
        video.stabilized = stabilized
    if status is not None:
        video.status = status
    if scale_m_per_px is not None:
        video.scale_m_per_px = scale_m_per_px
    if lane_config is not None:
        video.lane_config = lane_config
    if model_version is not None:
        video.model_version = model_version
    if tracker_name is not None:
        video.tracker_name = tracker_name
    return video


def _clear_results(session: Session, video_id: str) -> None:
    session.execute(delete(TrajectoryRow).where(TrajectoryRow.video_id == video_id))
    session.execute(delete(Vehicle).where(Vehicle.video_id == video_id))
    session.execute(delete(AnalyticsRow).where(AnalyticsRow.video_id == video_id))
    session.execute(delete(Event).where(Event.video_id == video_id))
    session.execute(delete(Report).where(Report.video_id == video_id))


def persist_results(
    session: Session,
    *,
    video_id: str,
    trajectories: Sequence[Trajectory],
    analytics: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    tracker_name: str | None = None,
    scale_m_per_px: float | None = None,
    lane_config: dict[str, Any] | None = None,
) -> Video:
    """Replace live result tables for ``video_id`` from generated artifacts."""
    video = require_video(session, video_id)
    fps = float(video.fps or analytics.get("fps") or 0.0)
    _clear_results(session, video_id)

    if model_version is not None:
        video.model_version = model_version
    if tracker_name is not None:
        video.tracker_name = tracker_name
    if scale_m_per_px is not None:
        video.scale_m_per_px = scale_m_per_px
    if lane_config is not None:
        video.lane_config = lane_config
    video.status = "completed"

    _insert_tracks(session, video_id, trajectories, fps=fps)
    units_blob = analytics.get("units") if isinstance(analytics.get("units"), Mapping) else {}
    unit_label = "m" if units_blob.get("labelled_as_physical") else "px"
    _insert_analytics_windows(session, video_id, analytics, fps=fps, units=unit_label)
    _insert_events(session, video_id, analytics, fps=fps)
    report = build_report(video, analytics)
    session.add(
        Report(
            video_id=video_id,
            payload=report,
            insights=list(analytics.get("insights") or []),
            analytics_json=dict(analytics),
            generated_at=utc_now_iso(),
        )
    )
    if diagnostics:
        session.add(
            EvaluationRun(
                eval_id=f"track_{video_id}_{uuid.uuid4().hex[:8]}",
                video_id=video_id,
                model_version=video.model_version,
                tracker_name=video.tracker_name,
                tracking_diagnostics=dict(diagnostics),
                created_at=utc_now_iso(),
            )
        )
    return video


def _insert_tracks(
    session: Session,
    video_id: str,
    trajectories: Sequence[Trajectory],
    *,
    fps: float,
) -> None:
    for traj in trajectories:
        class_id = int(traj.class_id)
        entry_t = (traj.entry_frame / fps) if fps > 0 and traj.entry_frame >= 0 else None
        exit_t = (traj.exit_frame / fps) if fps > 0 and traj.exit_frame >= 0 else None
        session.add(
            Vehicle(
                video_id=video_id,
                track_id=int(traj.track_id),
                class_id=class_id,
                vehicle_type=CLASS_NAMES.get(class_id, str(class_id)),
                entry_frame=traj.entry_frame if traj.entry_frame >= 0 else None,
                exit_frame=traj.exit_frame if traj.exit_frame >= 0 else None,
                entry_time=clock_time(entry_t) if entry_t is not None else None,
                exit_time=clock_time(exit_t) if exit_t is not None else None,
            )
        )
        for point in traj.points:
            session.add(
                TrajectoryRow(
                    video_id=video_id,
                    track_id=int(point.track_id),
                    frame=int(point.frame),
                    center_x=float(point.center_x),
                    center_y=float(point.center_y),
                    width=float(point.width),
                    height=float(point.height),
                    angle=float(point.angle),
                    confidence=float(point.confidence),
                    class_id=int(point.class_id),
                    lane=point.lane,
                )
            )


def _insert_analytics_windows(
    session: Session,
    video_id: str,
    analytics: Mapping[str, Any],
    *,
    fps: float,
    units: str,
) -> None:
    flow = analytics.get("flow") if isinstance(analytics.get("flow"), Mapping) else {}
    windows = flow.get("windows") if isinstance(flow, Mapping) else None
    if not isinstance(windows, list):
        return
    for window in windows:
        if not isinstance(window, Mapping):
            continue
        t0 = float(window.get("t0") or 0.0)
        t1 = float(window.get("t1") or 0.0)
        state = str(window.get("state") or "unknown")
        session.add(
            AnalyticsRow(
                video_id=video_id,
                frame_start=int(t0 * fps) if fps > 0 else None,
                frame_end=int(t1 * fps) if fps > 0 else None,
                vehicle_count=int(window["n_vehicles"]) if window.get("n_vehicles") is not None else None,
                average_speed=window.get("mean_speed"),
                density=window.get("density"),
                units=units,
                traffic_state=state,
                congestion_score=_STATE_SCORE.get(state),
            )
        )


def _insert_events(
    session: Session,
    video_id: str,
    analytics: Mapping[str, Any],
    *,
    fps: float,
) -> None:
    events = analytics.get("events")
    if not isinstance(events, list):
        return
    for i, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            continue
        event_type = str(raw.get("type") or "event")
        t = raw.get("t")
        frame = int(float(t) * fps) if t is not None and fps > 0 else None
        extra = {
            k: v
            for k, v in raw.items()
            if k
            not in {"type", "time", "video_id", "location", "severity", "t", "track_id"}
        }
        session.add(
            Event(
                event_id=f"{video_id}_{i:04d}_{event_type}",
                video_id=video_id,
                event_type=event_type,
                frame=frame,
                timestamp=str(raw.get("time")) if raw.get("time") is not None else None,
                zone_id=str(raw["location"]) if raw.get("location") else None,
                location=str(raw["location"]) if raw.get("location") else None,
                severity=str(raw["severity"]) if raw.get("severity") else None,
                track_id=int(raw["track_id"]) if raw.get("track_id") is not None else None,
                extra=extra or None,
            )
        )


def list_videos(session: Session) -> list[Video]:
    stmt = select(Video).order_by(Video.created_at.desc())
    return list(session.scalars(stmt))


def get_report(session: Session, video_id: str) -> Report | None:
    return session.get(Report, video_id)


def list_events(session: Session, video_id: str) -> list[Event]:
    stmt = select(Event).where(Event.video_id == video_id).order_by(Event.timestamp)
    return list(session.scalars(stmt))


def list_vehicles(session: Session, video_id: str) -> list[Vehicle]:
    stmt = select(Vehicle).where(Vehicle.video_id == video_id).order_by(Vehicle.track_id)
    return list(session.scalars(stmt))


def get_trajectory_points(
    session: Session,
    video_id: str,
    track_id: int,
    *,
    stride: int = 1,
) -> list[TrajectoryRow]:
    stmt = (
        select(TrajectoryRow)
        .where(
            TrajectoryRow.video_id == video_id,
            TrajectoryRow.track_id == track_id,
        )
        .order_by(TrajectoryRow.frame)
    )
    points = list(session.scalars(stmt))
    step = max(1, int(stride))
    return points[::step]


def list_map_tracks(
    session: Session,
    video_id: str,
    *,
    stride: int = 4,
    max_tracks: int = 120,
    max_points: int = 4000,
) -> dict[str, Any]:
    """Sampled plan-view polylines for the dashboard (not a full dump)."""
    vehicles = {row.track_id: row for row in list_vehicles(session, video_id)}
    stmt = (
        select(TrajectoryRow)
        .where(TrajectoryRow.video_id == video_id)
        .order_by(TrajectoryRow.track_id, TrajectoryRow.frame)
    )
    grouped: dict[int, list[TrajectoryRow]] = {}
    for row in session.scalars(stmt):
        grouped.setdefault(row.track_id, []).append(row)

    ranked = sorted(grouped, key=lambda tid: -len(grouped[tid]))[: max(1, int(max_tracks))]
    step = max(1, int(stride))
    tracks_out: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []
    budget = max(2, int(max_points))

    for track_id in ranked:
        if budget < 2:
            break
        raw = grouped[track_id]
        if not raw:
            continue
        sampled = raw[::step]
        if sampled[-1] is not raw[-1]:
            sampled = [*sampled, raw[-1]]
        if len(sampled) > budget:
            sampled = sampled[:budget]
        vehicle = vehicles.get(track_id)
        class_id = vehicle.class_id if vehicle is not None else sampled[0].class_id
        vehicle_type = vehicle.vehicle_type if vehicle is not None else None
        if not vehicle_type:
            vehicle_type = CLASS_NAMES.get(class_id if class_id is not None else -1, "unknown")
        points = []
        for point in sampled:
            points.append(
                {
                    "x": point.center_x,
                    "y": point.center_y,
                    "frame": point.frame,
                    "speed": point.speed,
                }
            )
            xs.append(point.center_x)
            ys.append(point.center_y)
        tracks_out.append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "vehicle_type": vehicle_type,
                "n_points": len(raw),
                "points": points,
            }
        )
        budget -= len(points)

    bounds = None
    if xs:
        bounds = {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}
    return {
        "video_id": video_id,
        "stride": step,
        "n_tracks": len(tracks_out),
        "n_points": sum(len(t["points"]) for t in tracks_out),
        "bounds": bounds,
        "tracks": tracks_out,
    }


def list_analytics_rows(session: Session, video_id: str) -> list[AnalyticsRow]:
    stmt = (
        select(AnalyticsRow)
        .where(AnalyticsRow.video_id == video_id)
        .order_by(AnalyticsRow.id)
    )
    return list(session.scalars(stmt))
