"""Rule-based events: stopped vehicles, sudden congestion, queue spillback."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from analytics.types import (
    FlowWindow,
    Sample,
    SceneContext,
    TrafficEvent,
    clock_time,
)


def _stopped_threshold(context: SceneContext, cfg: Mapping[str, Any]) -> float:
    if context.units.labelled_as_physical:
        return float(cfg.get("stopped_speed_m_s") or 0.5)
    return float(cfg.get("stopped_speed_px_s") or 8.0)


def _severity(value: float, *, medium: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _location(sample: Sample) -> str:
    return sample.zone or sample.lane or f"{sample.x:.0f},{sample.y:.0f}"


def detect_events(
    samples: Sequence[Sample],
    context: SceneContext,
    cfg: Mapping[str, Any],
    *,
    windows: Sequence[FlowWindow],
    zone_windows: Sequence[dict[str, Any]] | None = None,
) -> list[TrafficEvent]:
    """Emit FR-EVT records from kinematics + flow windows."""
    events: list[TrafficEvent] = []
    events.extend(_stopped_vehicles(samples, context, cfg))
    events.extend(_sudden_congestion(windows, context, cfg))
    events.extend(_spillback(zone_windows or [], context, cfg))
    events.sort(key=lambda e: (e.t or 0.0, e.type))
    return events


def _stopped_vehicles(
    samples: Sequence[Sample],
    context: SceneContext,
    cfg: Mapping[str, Any],
) -> list[TrafficEvent]:
    stopped = _stopped_threshold(context, cfg)
    min_s = float(cfg.get("stopped_min_seconds") or 1.5)
    by_track: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_track[sample.track_id].append(sample)

    events: list[TrafficEvent] = []
    for track_id, rows in by_track.items():
        rows = sorted(rows, key=lambda s: s.t)
        run: list[Sample] = []

        def flush() -> None:
            if len(run) < 2:
                return
            dur = run[-1].t - run[0].t
            if dur < min_s:
                return
            loc = _location(run[0])
            events.append(
                TrafficEvent(
                    type="stopped_vehicle",
                    time=clock_time(run[0].t),
                    video_id=context.video_id,
                    location=loc,
                    severity=_severity(dur, medium=min_s * 2, high=min_s * 4),
                    t=run[0].t,
                    track_id=track_id,
                    extra={"duration_s": dur},
                )
            )

        for sample in rows:
            if sample.speed is not None and sample.speed <= stopped:
                run.append(sample)
            else:
                flush()
                run = []
        flush()
    return events


def _sudden_congestion(
    windows: Sequence[FlowWindow],
    context: SceneContext,
    cfg: Mapping[str, Any],
) -> list[TrafficEvent]:
    drop = float(cfg.get("sudden_speed_drop") or 0.4)
    events: list[TrafficEvent] = []
    for prev, curr in zip(windows, windows[1:]):
        if prev.mean_speed is None or curr.mean_speed is None:
            continue
        if prev.mean_speed <= 0:
            continue
        rel = (prev.mean_speed - curr.mean_speed) / prev.mean_speed
        if rel < drop:
            continue
        if curr.n_vehicles < prev.n_vehicles:
            continue
        events.append(
            TrafficEvent(
                type="sudden_congestion",
                time=clock_time(curr.t0),
                video_id=context.video_id,
                location="scene",
                severity=_severity(rel, medium=drop, high=min(0.75, drop + 0.2)),
                t=curr.t0,
                extra={
                    "speed_drop": rel,
                    "from_speed": prev.mean_speed,
                    "to_speed": curr.mean_speed,
                },
            )
        )
    return events


def _spillback(
    zone_summaries: Sequence[dict[str, Any]],
    context: SceneContext,
    cfg: Mapping[str, Any],
) -> list[TrafficEvent]:
    """Queue growing across consecutive windows in the same zone (FR-EVT-003)."""
    queue_cut = float(cfg.get("queue_fraction") or 0.5)
    events: list[TrafficEvent] = []
    for zone in zone_summaries:
        windows = zone.get("windows") or []
        zid = str(zone.get("zone") or "zone")
        for prev, curr in zip(windows, windows[1:]):
            prev_q = float(prev.get("queue_fraction") or 0.0)
            curr_q = float(curr.get("queue_fraction") or 0.0)
            prev_n = int(prev.get("n_vehicles") or 0)
            curr_n = int(curr.get("n_vehicles") or 0)
            if curr_q < queue_cut:
                continue
            if curr_n <= prev_n and curr_q <= prev_q:
                continue
            events.append(
                TrafficEvent(
                    type="queue_spillback",
                    time=clock_time(float(curr.get("t0") or 0.0)),
                    video_id=context.video_id,
                    location=zid,
                    severity=_severity(curr_q, medium=queue_cut, high=0.8),
                    t=float(curr.get("t0") or 0.0),
                    extra={"queue_fraction": curr_q, "n_vehicles": curr_n},
                )
            )
    return events
