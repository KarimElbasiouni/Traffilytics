"""Traffic flow characterization (FR-FLOW-001 … FR-FLOW-005)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from analytics.kinematics import samples_in_window, window_bounds
from analytics.types import FlowResult, FlowWindow, Sample, SceneContext, mean, quantile


def _state(
    mean_speed: float | None,
    *,
    free_flow: float | None,
    stopped: float,
    congested_ratio: float,
) -> str:
    if mean_speed is None:
        return "unknown"
    if mean_speed <= stopped:
        return "queued"
    if free_flow is None or free_flow <= 0:
        return "unknown"
    if mean_speed < congested_ratio * free_flow:
        return "congested"
    return "free"


def characterize_flow(
    samples: Sequence[Sample],
    context: SceneContext,
    cfg: Mapping[str, Any],
    *,
    n_tracks: int,
) -> FlowResult:
    """Volume, speed, density, congestion state, and flow–density windows."""
    window_seconds = float(cfg.get("window_seconds") or 2.0)
    quantile_q = float(cfg.get("free_speed_quantile") or 0.85)
    congested_ratio = float(cfg.get("congested_speed_ratio") or 0.5)
    if context.units.labelled_as_physical:
        stopped = float(cfg.get("stopped_speed_m_s") or 0.5)
    else:
        stopped = float(cfg.get("stopped_speed_px_s") or 8.0)

    speeds = [s.speed for s in samples if s.speed is not None]
    free_flow = quantile(speeds, quantile_q)
    clip_mean = mean(speeds)

    if not samples:
        return FlowResult(
            vehicles_per_minute=0.0,
            mean_speed=None,
            mean_density=None,
            free_flow_speed=free_flow,
            windows=[],
        )

    t_min = min(s.t for s in samples)
    t_max = max(s.t for s in samples)
    duration = max(t_max - t_min, 1.0 / max(context.fps, 1e-6))
    vehicles_per_minute = n_tracks * (60.0 / duration)

    area, _ = context.units.area(context.width, context.height)

    windows: list[FlowWindow] = []
    densities: list[float] = []
    for t0, t1 in window_bounds(t_min, t_max, window_seconds):
        in_window = samples_in_window(samples, t0, t1, t_max=t_max)
        ids = {s.track_id for s in in_window}
        n_vehicles = len(ids)
        win_dur = max(t1 - t0, 1e-6)
        volume_per_min = n_vehicles * (60.0 / win_dur)
        win_speeds = [s.speed for s in in_window if s.speed is not None]
        mean_speed = mean(win_speeds)

        by_frame: dict[int, set[int]] = defaultdict(set)
        for s in in_window:
            by_frame[s.frame].add(s.track_id)
        occupancy = mean([float(len(v)) for v in by_frame.values()]) or 0.0
        density = (occupancy / area) if area else None
        flow = None
        if density is not None and mean_speed is not None:
            flow = density * mean_speed
        if density is not None:
            densities.append(density)
        windows.append(
            FlowWindow(
                t0=t0,
                t1=t1,
                n_vehicles=n_vehicles,
                volume_per_min=volume_per_min,
                mean_speed=mean_speed,
                density=density,
                occupancy_mean=occupancy,
                flow=flow,
                state=_state(
                    mean_speed,
                    free_flow=free_flow,
                    stopped=stopped,
                    congested_ratio=congested_ratio,
                ),
            )
        )

    return FlowResult(
        vehicles_per_minute=vehicles_per_minute,
        mean_speed=clip_mean,
        mean_density=mean(densities),
        free_flow_speed=free_flow,
        windows=windows,
    )
