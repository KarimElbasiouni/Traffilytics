"""Bottleneck identification over user-defined zones (FR-BTN-001 … FR-BTN-004)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from analytics.kinematics import samples_in_window
from analytics.types import BottleneckResult, Sample, SceneContext, ZoneWindow, mean


def _stopped_threshold(context: SceneContext, cfg: Mapping[str, Any]) -> float:
    if context.units.labelled_as_physical:
        return float(cfg.get("stopped_speed_m_s") or 0.5)
    return float(cfg.get("stopped_speed_px_s") or 8.0)


def _cause(*, queue_fraction: float, queue_cut: float, speed: float | None, scene_speed: float | None) -> str:
    if queue_fraction >= queue_cut:
        return "queue_formation"
    if speed is not None and scene_speed and scene_speed > 0 and speed < 0.6 * scene_speed:
        return "speed_reduction"
    return "accumulation"


def _heatmap(samples: Sequence[Sample], context: SceneContext, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = max(1, int(cfg.get("heatmap_rows") or 12))
    cols = max(1, int(cfg.get("heatmap_cols") or 12))
    if context.width and context.height:
        x0, y0, x1, y1 = 0.0, 0.0, float(context.width), float(context.height)
    elif samples:
        xs = [s.x for s in samples]
        ys = [s.y for s in samples]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
    else:
        return []

    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows
    counts: dict[tuple[int, int], list[float]] = defaultdict(list)
    for sample in samples:
        c = min(cols - 1, max(0, int((sample.x - x0) / cell_w))) if cell_w else 0
        r = min(rows - 1, max(0, int((sample.y - y0) / cell_h))) if cell_h else 0
        counts[(r, c)].append(sample.speed if sample.speed is not None else 0.0)

    cells: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            speeds = counts.get((r, c)) or []
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "x0": x0 + c * cell_w,
                    "y0": y0 + r * cell_h,
                    "x1": x0 + (c + 1) * cell_w,
                    "y1": y0 + (r + 1) * cell_h,
                    "n_points": len(speeds),
                    "mean_speed": mean(speeds),
                }
            )
    return cells


def identify_bottlenecks(
    samples: Sequence[Sample],
    context: SceneContext,
    cfg: Mapping[str, Any],
    *,
    window_bounds: Sequence[tuple[float, float]],
    scene_mean_speed: float | None,
) -> BottleneckResult:
    """Score zones by speed drop, accumulation, and queue; emit a coarse heatmap."""
    assigner = context.assigner
    zone_ids = [z.id for z in assigner.zones] if assigner is not None else []
    heatmap = _heatmap(samples, context, cfg)
    if not zone_ids:
        return BottleneckResult(
            configured=False,
            heatmap=heatmap,
            note="No zones in the lane/zone JSON; bottleneck location skipped (FR-BTN-001).",
        )

    stopped = _stopped_threshold(context, cfg)
    queue_cut = float(cfg.get("queue_fraction") or 0.5)
    zone_rows: dict[str, list[ZoneWindow]] = {zid: [] for zid in zone_ids}

    t_max = max((s.t for s in samples), default=0.0)
    for t0, t1 in window_bounds:
        in_window = samples_in_window(samples, t0, t1, t_max=t_max)
        for zid in zone_ids:
            z_samples = [s for s in in_window if s.zone == zid]
            n_vehicles = len({s.track_id for s in z_samples})
            z_speeds = [s.speed for s in z_samples if s.speed is not None]
            mean_speed = mean(z_speeds)
            if z_speeds:
                queued = sum(1 for v in z_speeds if v <= stopped) / len(z_speeds)
            else:
                queued = 0.0
            speed_drop = 0.0
            if mean_speed is not None and scene_mean_speed and scene_mean_speed > 0:
                speed_drop = max(0.0, 1.0 - (mean_speed / scene_mean_speed))
            accum = float(n_vehicles)
            score = speed_drop + queued + accum / max(1.0, float(len({s.track_id for s in in_window}) or 1))
            zone_rows[zid].append(
                ZoneWindow(
                    zone=zid,
                    t0=t0,
                    t1=t1,
                    n_vehicles=n_vehicles,
                    mean_speed=mean_speed,
                    queue_fraction=queued,
                    score=score,
                )
            )

    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for zid, windows in zone_rows.items():
        scores = [w.score for w in windows]
        mean_score = mean(scores) or 0.0
        mean_q = mean([w.queue_fraction for w in windows]) or 0.0
        z_speed = mean([w.mean_speed for w in windows if w.mean_speed is not None])
        cause = _cause(queue_fraction=mean_q, queue_cut=queue_cut, speed=z_speed, scene_speed=scene_mean_speed)
        row = {
            "zone": zid,
            "mean_score": mean_score,
            "mean_speed": z_speed,
            "mean_queue_fraction": mean_q,
            "cause": cause,
            "windows": [w.to_dict() for w in windows],
        }
        summaries.append(row)
        if best is None or mean_score > float(best.get("score") or -1):
            best = {"zone": zid, "cause": cause, "score": mean_score}

    primary = best if best and float(best["score"]) > 0 else None
    return BottleneckResult(configured=True, primary=primary, zones=summaries, heatmap=heatmap)
