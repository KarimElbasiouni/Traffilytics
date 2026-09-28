"""Analytics query endpoints plus overview."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.api.deps import get_db, get_settings
from backend.api.errors import ApiError
from backend.api.routes.videos import _video_out, load_analytics_json
from backend.database.persist import get_video, list_events
from backend.database.session import Database
from backend.services.settings import Settings

router = APIRouter()


def _units(analytics: dict[str, Any]) -> dict[str, Any]:
    units = analytics.get("units")
    return dict(units) if isinstance(units, dict) else {}


@router.get("/videos/{video_id}/analytics/flow")
def analytics_flow(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    analytics = load_analytics_json(db, settings, video_id)
    flow = analytics.get("flow") if isinstance(analytics.get("flow"), dict) else {}
    return {
        "video_id": video_id,
        "units": _units(analytics),
        "flow": flow,
    }


@router.get("/videos/{video_id}/analytics/flow-density")
def analytics_flow_density(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    analytics = load_analytics_json(db, settings, video_id)
    flow = analytics.get("flow") if isinstance(analytics.get("flow"), dict) else {}
    return {
        "video_id": video_id,
        "units": _units(analytics),
        "flow_density": flow.get("flow_density") or [],
    }


@router.get("/videos/{video_id}/analytics/bottlenecks")
def analytics_bottlenecks(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    analytics = load_analytics_json(db, settings, video_id)
    bottleneck = (
        analytics.get("bottleneck") if isinstance(analytics.get("bottleneck"), dict) else {}
    )
    return {
        "video_id": video_id,
        "units": _units(analytics),
        "bottleneck": bottleneck,
    }


@router.get("/videos/{video_id}/analytics/imbalance")
def analytics_imbalance(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    analytics = load_analytics_json(db, settings, video_id)
    imbalance = (
        analytics.get("imbalance") if isinstance(analytics.get("imbalance"), dict) else {}
    )
    return {
        "video_id": video_id,
        "units": _units(analytics),
        "imbalance": imbalance,
    }


@router.get("/videos/{video_id}/analytics/heatmap")
def analytics_heatmap(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    analytics = load_analytics_json(db, settings, video_id)
    bottleneck = (
        analytics.get("bottleneck") if isinstance(analytics.get("bottleneck"), dict) else {}
    )
    return {
        "video_id": video_id,
        "units": _units(analytics),
        "heatmap": bottleneck.get("heatmap") or [],
    }


@router.get("/videos/{video_id}/analytics/micro")
def analytics_micro(video_id: str, db: Database = Depends(get_db)) -> dict[str, Any]:
    with db.session_scope() as session:
        if get_video(session, video_id) is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
    return {
        "video_id": video_id,
        "available": False,
        "note": "Optional LC/TTC micro-analytics are not implemented (Epic 4).",
    }


@router.get("/videos/{video_id}/overview")
def overview(
    video_id: str,
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    analytics = load_analytics_json(db, settings, video_id)
    flow = analytics.get("flow") if isinstance(analytics.get("flow"), dict) else {}
    windows = flow.get("windows") if isinstance(flow.get("windows"), list) else []
    states = [str(w.get("state")) for w in windows if isinstance(w, dict) and w.get("state")]
    traffic_state = max(set(states), key=states.count) if states else "unknown"
    events = analytics.get("events") if isinstance(analytics.get("events"), list) else []
    major = [
        e
        for e in events
        if isinstance(e, dict) and str(e.get("severity") or "").lower() in {"high", "medium"}
    ]
    with db.session_scope() as session:
        video = get_video(session, video_id)
        if video is None:
            raise ApiError(404, "not_found", f"Unknown video_id: {video_id}")
        stored_events = list_events(session, video_id)
        video_payload = _video_out(video)
    return {
        **video_payload,
        "n_vehicles": analytics.get("n_trajectories") or 0,
        "traffic_state": traffic_state,
        "mean_speed": flow.get("mean_speed"),
        "vehicles_per_minute": flow.get("vehicles_per_minute"),
        "units": _units(analytics),
        "major_events": major[:10] if major else events[:5],
        "n_events": len(stored_events) if stored_events else len(events),
        "insights": analytics.get("insights") or [],
    }
