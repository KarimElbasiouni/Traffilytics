"""Automated transportation analysis summaries (FR-RPT-001)."""

from __future__ import annotations

import subprocess
from typing import Any, Mapping

from backend.database.models import Video, utc_now_iso

_REPO = "https://github.com/KarimElbasiouni/Traffilytics"
_OBB_V1 = "1100b33da026c692a5595b3628cf5d5c75a3366e"
_WEIGHTS_URL = f"{_REPO}/releases/download/obb-v1/your_obb.pt"


def _source_commit() -> str:
    """Commit of this checkout. Falls back to the obb-v1 tag when git is absent."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return _OBB_V1
    return sha or _OBB_V1


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
            "source_commit": _source_commit(),
            "source_commit_url": f"{_REPO}/commit/{_source_commit()}",
            "release_tag": "obb-v1",
            "release_url": f"{_REPO}/releases/tag/obb-v1",
            "weights_url": _WEIGHTS_URL,
        },
    }


def _majority(values: list[str]) -> str | None:
    if not values:
        return None
    return max(set(values), key=values.count)
