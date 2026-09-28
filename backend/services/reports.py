"""Automated transportation analysis summaries (FR-RPT-001)."""

from __future__ import annotations

from typing import Any, Mapping

from backend.database.models import Video, utc_now_iso


def build_report(video: Video, analytics: Mapping[str, Any]) -> dict[str, Any]:
    """Build a dashboard/report payload from stored analytics JSON."""
    flow = analytics.get("flow") if isinstance(analytics.get("flow"), Mapping) else {}
    bottleneck = (
        analytics.get("bottleneck") if isinstance(analytics.get("bottleneck"), Mapping) else {}
    )
    imbalance = (
        analytics.get("imbalance") if isinstance(analytics.get("imbalance"), Mapping) else {}
    )
    events = analytics.get("events") if isinstance(analytics.get("events"), list) else []
    insights = analytics.get("insights") if isinstance(analytics.get("insights"), list) else []
    units = analytics.get("units") if isinstance(analytics.get("units"), Mapping) else {}

    windows = flow.get("windows") if isinstance(flow, Mapping) else None
    states = [
        str(w.get("state"))
        for w in (windows or [])
        if isinstance(w, Mapping) and w.get("state")
    ]
    traffic_state = _majority(states) or "unknown"

    findings: list[str] = []
    for item in insights:
        if isinstance(item, Mapping) and item.get("text"):
            findings.append(str(item["text"]))
        elif isinstance(item, str):
            findings.append(item)

    high_events = [
        e
        for e in events
        if isinstance(e, Mapping) and str(e.get("severity") or "").lower() == "high"
    ]

    return {
        "video_id": video.video_id,
        "title": f"Transportation analysis — {video.video_id}",
        "generated_at": utc_now_iso(),
        "site": video.site,
        "source": video.source,
        "model_version": video.model_version,
        "tracker_name": video.tracker_name,
        "status": video.status,
        "units": dict(units),
        "overview": {
            "n_vehicles": analytics.get("n_trajectories") or 0,
            "traffic_state": traffic_state,
            "vehicles_per_minute": flow.get("vehicles_per_minute") if isinstance(flow, Mapping) else None,
            "mean_speed": flow.get("mean_speed") if isinstance(flow, Mapping) else None,
            "mean_density": flow.get("mean_density") if isinstance(flow, Mapping) else None,
            "n_events": len(events),
            "n_high_severity": len(high_events),
        },
        "flow": dict(flow) if isinstance(flow, Mapping) else {},
        "bottleneck": dict(bottleneck) if isinstance(bottleneck, Mapping) else {},
        "imbalance": dict(imbalance) if isinstance(imbalance, Mapping) else {},
        "events": list(events),
        "insights": list(insights),
        "findings": findings,
        "attribution": {
            "uav_obb": (
                "UAV-OBB (Ahmad, Fengjun, Bibi & Slaman Pathan, 2026), "
                "Mendeley Data V3, doi:10.17632/6snrjwcpkh.3, CC BY 4.0. "
                "Imagery shown from that dataset must carry this citation."
            ),
            "licence": "Traffilytics is AGPL-3.0-only.",
        },
    }


def _majority(values: list[str]) -> str | None:
    if not values:
        return None
    return max(set(values), key=values.count)
