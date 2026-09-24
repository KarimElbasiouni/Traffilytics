"""AnalyticsEngine facade: trajectories in, analytics/events/insights out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from analytics.bottleneck.bottleneck import identify_bottlenecks
from analytics.events.events import detect_events
from analytics.imbalance.imbalance import analyze_imbalance
from analytics.insights.insights import generate_insights
from analytics.kinematics import build_samples
from analytics.traffic_flow.flow import characterize_flow
from analytics.types import (
    DEFAULT_ANALYTICS_NAME,
    AnalyticsError,
    AnalyticsReport,
    SceneContext,
)
from computer_vision.trajectories.types import Trajectory

DEFAULT_ANALYTICS_CFG: dict[str, Any] = {
    "window_seconds": 2.0,
    "pixels_per_metre": None,
    "stopped_speed_px_s": 8.0,
    "stopped_speed_m_s": 0.5,
    "stopped_min_seconds": 1.5,
    "free_speed_quantile": 0.85,
    "congested_speed_ratio": 0.5,
    "sudden_speed_drop": 0.4,
    "queue_fraction": 0.5,
    "heatmap_rows": 12,
    "heatmap_cols": 12,
}


def merge_analytics_cfg(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill analytics YAML keys with defaults; ignore unknown keys."""
    cfg = dict(DEFAULT_ANALYTICS_CFG)
    if not raw:
        return cfg
    for key in DEFAULT_ANALYTICS_CFG:
        if key in raw and raw[key] is not None:
            cfg[key] = raw[key]
    return cfg


@dataclass
class AnalyticsEngine:
    """Plan-then-run facade over flow, bottleneck, imbalance, events, insights."""

    cfg: dict[str, Any]

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = merge_analytics_cfg(cfg)

    def analyze(
        self,
        trajectories: Sequence[Trajectory],
        *,
        context: SceneContext,
    ) -> AnalyticsReport:
        if context.fps <= 0:
            raise AnalyticsError(f"fps must be > 0, got {context.fps}")
        samples = build_samples(trajectories, context)
        flow = characterize_flow(
            samples, context, self.cfg, n_tracks=len(trajectories)
        )
        window_bounds = [(w.t0, w.t1) for w in flow.windows]
        bottleneck = identify_bottlenecks(
            samples,
            context,
            self.cfg,
            window_bounds=window_bounds,
            scene_mean_speed=flow.mean_speed,
        )
        imbalance = analyze_imbalance(samples, context, self.cfg)
        events = detect_events(
            samples,
            context,
            self.cfg,
            windows=flow.windows,
            zone_windows=bottleneck.zones,
        )
        insights = generate_insights(
            video_id=context.video_id,
            n_trajectories=len(trajectories),
            units_note=context.units.to_dict()["note"],
            speed_unit=context.units.speed_unit,
            flow=flow,
            bottleneck=bottleneck,
            imbalance=imbalance,
            events=events,
        )
        return AnalyticsReport(
            video_id=context.video_id,
            units=context.units,
            flow=flow,
            bottleneck=bottleneck,
            imbalance=imbalance,
            events=events,
            insights=insights,
            fps=context.fps,
            n_trajectories=len(trajectories),
            source=context.source,
            tracker=context.tracker,
            config=dict(self.cfg),
        )

    def write_json(self, report: AnalyticsReport, dest: str | Path) -> Path:
        """Write ``analytics.json`` (flow, bottleneck, imbalance, events, insights)."""
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        return dest_path
