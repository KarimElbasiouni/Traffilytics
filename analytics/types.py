"""Shared analytics records (Epic 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from computer_vision.trajectories.lanes import LaneAssigner


class AnalyticsError(Exception):
    """Raised when analytics cannot run (missing input, bad fps, …)."""

    exit_code = 1


DEFAULT_ANALYTICS_NAME = "analytics.json"


@dataclass(frozen=True)
class UnitSystem:
    """Speed/density units for one run (FR-FLOW-005, NFR-ACC-006)."""

    pixels_per_metre: float | None = None

    @property
    def labelled_as_physical(self) -> bool:
        return self.pixels_per_metre is not None and self.pixels_per_metre > 0

    @property
    def speed_unit(self) -> str:
        return "m/s" if self.labelled_as_physical else "px/s"

    @property
    def density_unit(self) -> str:
        return "vehicles/m^2" if self.labelled_as_physical else "vehicles/px^2"

    def speed_from_pixels(self, distance_px: float, dt_s: float) -> float:
        """Convert a pixel displacement over ``dt_s`` into the run's speed unit."""
        if dt_s <= 0:
            return 0.0
        px_s = float(distance_px) / dt_s
        if self.labelled_as_physical:
            return px_s / float(self.pixels_per_metre)
        return px_s

    def area(self, width: float | None, height: float | None) -> tuple[float | None, str]:
        """Return (area, unit). Area is ``None`` when the frame size is unknown."""
        if width is None or height is None or width <= 0 or height <= 0:
            return None, self.density_unit
        px_area = float(width) * float(height)
        if self.labelled_as_physical:
            scale = float(self.pixels_per_metre)
            return px_area / (scale * scale), self.density_unit
        return px_area, self.density_unit

    def to_dict(self) -> dict[str, Any]:
        note = (
            "physical units from pixels_per_metre"
            if self.labelled_as_physical
            else "pixel-based — no ground scale"
        )
        return {
            "speed": self.speed_unit,
            "density": self.density_unit,
            "pixels_per_metre": self.pixels_per_metre,
            "labelled_as_physical": self.labelled_as_physical,
            "note": note,
        }


@dataclass
class SceneContext:
    """Per-video inputs analytics needs besides the trajectory list."""

    video_id: str
    fps: float
    width: int | None = None
    height: int | None = None
    pixels_per_metre: float | None = None
    assigner: LaneAssigner | None = None
    source: str | None = None
    tracker: str | None = None

    @property
    def units(self) -> UnitSystem:
        return UnitSystem(pixels_per_metre=self.pixels_per_metre)


@dataclass
class Sample:
    """One trajectory point with derived speed/heading/zone (working set)."""

    track_id: int
    frame: int
    t: float
    x: float
    y: float
    speed: float | None
    heading: float | None
    lane: str | None
    zone: str | None
    class_id: int
    video_id: str


@dataclass
class FlowWindow:
    t0: float
    t1: float
    n_vehicles: int
    volume_per_min: float
    mean_speed: float | None
    density: float | None
    occupancy_mean: float
    flow: float | None
    state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "n_vehicles": self.n_vehicles,
            "volume_per_min": self.volume_per_min,
            "mean_speed": self.mean_speed,
            "density": self.density,
            "occupancy_mean": self.occupancy_mean,
            "flow": self.flow,
            "state": self.state,
        }


@dataclass
class FlowResult:
    vehicles_per_minute: float
    mean_speed: float | None
    mean_density: float | None
    free_flow_speed: float | None
    windows: list[FlowWindow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicles_per_minute": self.vehicles_per_minute,
            "mean_speed": self.mean_speed,
            "mean_density": self.mean_density,
            "free_flow_speed": self.free_flow_speed,
            "windows": [w.to_dict() for w in self.windows],
            "flow_density": [
                {
                    "density": w.density,
                    "flow": w.flow,
                    "mean_speed": w.mean_speed,
                    "state": w.state,
                    "t0": w.t0,
                    "t1": w.t1,
                }
                for w in self.windows
            ],
        }


@dataclass
class ZoneWindow:
    zone: str
    t0: float
    t1: float
    n_vehicles: int
    mean_speed: float | None
    queue_fraction: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "t0": self.t0,
            "t1": self.t1,
            "n_vehicles": self.n_vehicles,
            "mean_speed": self.mean_speed,
            "queue_fraction": self.queue_fraction,
            "score": self.score,
        }


@dataclass
class BottleneckResult:
    configured: bool
    primary: dict[str, Any] | None = None
    zones: list[dict[str, Any]] = field(default_factory=list)
    heatmap: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "configured": self.configured,
            "primary": self.primary,
            "zones": self.zones,
            "heatmap": self.heatmap,
        }
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class LaneShare:
    lane: str
    n_tracks: int
    share: float
    mean_heading: float | None
    mean_speed: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "n_tracks": self.n_tracks,
            "share": self.share,
            "mean_heading": self.mean_heading,
            "mean_speed": self.mean_speed,
        }


@dataclass
class ImbalanceResult:
    configured: bool
    lanes: list[LaneShare] = field(default_factory=list)
    imbalance_index: float | None = None
    dominant_lane: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "configured": self.configured,
            "lanes": [row.to_dict() for row in self.lanes],
            "imbalance_index": self.imbalance_index,
            "dominant_lane": self.dominant_lane,
        }
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class TrafficEvent:
    type: str
    time: str
    video_id: str
    location: str
    severity: str
    t: float | None = None
    track_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "type": self.type,
            "time": self.time,
            "video_id": self.video_id,
            "location": self.location,
            "severity": self.severity,
        }
        if self.t is not None:
            row["t"] = self.t
        if self.track_id is not None:
            row["track_id"] = self.track_id
        row.update(self.extra)
        return row


@dataclass
class Insight:
    id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass
class AnalyticsReport:
    video_id: str
    units: UnitSystem
    flow: FlowResult
    bottleneck: BottleneckResult
    imbalance: ImbalanceResult
    events: list[TrafficEvent] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    fps: float = 0.0
    n_trajectories: int = 0
    source: str | None = None
    tracker: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source": self.source,
            "tracker": self.tracker,
            "fps": self.fps,
            "n_trajectories": self.n_trajectories,
            "units": self.units.to_dict(),
            "flow": self.flow.to_dict(),
            "bottleneck": self.bottleneck.to_dict(),
            "imbalance": self.imbalance.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "insights": [i.to_dict() for i in self.insights],
            "n_events": len(self.events),
            "config": self.config,
        }


def clock_time(t: float) -> str:
    """Format seconds as ``HH:MM:SS`` (FR-EVT example)."""
    total = max(0, int(round(float(t))))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    q = min(1.0, max(0.0, float(q)))
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
