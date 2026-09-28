"""Events, insights, and automated reports."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.api.deps import get_db, get_settings
from backend.api.errors import ApiError
from backend.api.routes.videos import load_analytics_json
from backend.database.models import Event, Report
from backend.database.persist import get_report, get_video, list_events
from backend.database.session import Database
from backend.services.reports import build_report
from backend.services.settings import Settings

router = APIRouter()


def _event_out(event: Event) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": event.event_id,
        "video_id": event.video_id,
        "event_type": event.event_type,
        "frame": event.frame,
        "timestamp": event.timestamp,
        "zone_id": event.zone_id,
        "location": event.location,
        "severity": event.severity,
        "track_id": event.track_id,
    }
    if event.extra:
        row.update(event.extra)
    return row


@router.get("/videos/{video_id}/events")
def events_index(video_id: str, db: Database = Depends(get_db)) -> dict[str, Any]:
    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        rows = list_events(session, video_id)
        return {"video_id": video_id, "n_events": len(rows), "events": [_event_out(e) for e in rows]}


@router.get("/videos/{video_id}/events/{event_id}")
def event_detail(video_id: str, event_id: str, db: Database = Depends(get_db)) -> dict[str, Any]:
    with db.session_scope() as session:
        event = session.get(Event, event_id)
        if event is None or event.video_id != video_id:
            raise ApiError(404, "not_found", f"Unknown event_id: {event_id}")
        return _event_out(event)


@router.get("/videos/{video_id}/insights")
def insights(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    with db.session_scope() as session:
        report = get_report(session, video_id)
        if report is not None:
            return {"video_id": video_id, "insights": report.insights or []}
    analytics = load_analytics_json(db, settings, video_id)
    return {"video_id": video_id, "insights": analytics.get("insights") or []}


@router.get("/videos/{video_id}/reports")
def reports(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        stored = session.get(Report, video_id)
        if stored is not None and stored.payload:
            return stored.payload
        analytics_json = stored.analytics_json if stored is not None else None
    if not analytics_json:
        analytics_json = load_analytics_json(db, settings, video_id)
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        return build_report(video, analytics_json)
