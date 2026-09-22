"""User-defined lane and zone polygons (FR-TRK-006).

UAV-OBB ships no lane topology. A per-video JSON lists named polygons; the
assigner maps a trajectory point's center into the first containing lane
(file order). Points outside every polygon keep ``lane: null``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from computer_vision.trajectories.types import Trajectory, TrajectoryPoint

DEFAULT_LANES_DIR = Path("configs/lanes")


class LaneConfigError(Exception):
    """Raised when a lane JSON is missing, malformed, or has unusable polygons."""

    exit_code = 1


@dataclass(frozen=True)
class NamedPolygon:
    """A named closed polygon in pixel coordinates."""

    id: str
    polygon: tuple[tuple[float, float], ...]

    def contour(self) -> np.ndarray:
        """``(N, 1, 2)`` float32 contour for ``cv2.pointPolygonTest``."""
        return np.array(self.polygon, dtype=np.float32).reshape((-1, 1, 2))

    def contains(self, x: float, y: float) -> bool:
        """True if ``(x, y)`` is inside or on the boundary."""
        return float(cv2.pointPolygonTest(self.contour(), (float(x), float(y)), False)) >= 0.0


def _as_points(raw: Any, *, name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise LaneConfigError(f"{name}: polygon needs at least 3 [x, y] vertices")
    pts: list[tuple[float, float]] = []
    for item in raw:
        try:
            x, y = item
        except (TypeError, ValueError) as exc:
            raise LaneConfigError(f"{name}: each vertex must be [x, y], got {item!r}") from exc
        pts.append((float(x), float(y)))
    return tuple(pts)


def _parse_named(entries: Any, *, kind: str) -> list[NamedPolygon]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise LaneConfigError(f"{kind} must be a list")
    polygons: list[NamedPolygon] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise LaneConfigError(f"{kind}[{i}] must be an object with id and polygon")
        ident = str(entry.get("id") or "").strip()
        if not ident:
            raise LaneConfigError(f"{kind}[{i}] is missing id")
        pts = _as_points(entry.get("polygon"), name=f"{kind} {ident}")
        polygons.append(NamedPolygon(id=ident, polygon=pts))
    return polygons


@dataclass
class LaneAssigner:
    """Point-in-polygon lookup for one video's lane/zone config."""

    video_id: str | None
    lanes: list[NamedPolygon]
    zones: list[NamedPolygon]
    source: Path | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> LaneAssigner:
        """Load ``configs/lanes/<video_id>.json`` (or any explicit JSON path)."""
        dest = Path(path)
        if not dest.is_file():
            raise LaneConfigError(f"Lane config not found: {dest}")
        try:
            payload = json.loads(dest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LaneConfigError(f"Invalid lane JSON: {dest} ({exc})") from exc
        if not isinstance(payload, dict):
            raise LaneConfigError(f"Lane config must be an object: {dest}")
        return cls(
            video_id=str(payload["video_id"]) if payload.get("video_id") else None,
            lanes=_parse_named(payload.get("lanes"), kind="lanes"),
            zones=_parse_named(payload.get("zones"), kind="zones"),
            source=dest,
        )

    @classmethod
    def discover(
        cls,
        video_id: str,
        *,
        lanes_dir: str | Path = DEFAULT_LANES_DIR,
        repo_root: str | Path | None = None,
    ) -> LaneAssigner | None:
        """Return a loader for ``<lanes_dir>/<video_id>.json`` if that file exists."""
        root = Path(repo_root) if repo_root is not None else Path()
        directory = Path(lanes_dir)
        if not directory.is_absolute():
            directory = (root / directory).resolve()
        path = directory / f"{video_id}.json"
        if not path.is_file():
            return None
        return cls.from_path(path)

    def assign_lane(self, x: float, y: float) -> str | None:
        """First lane polygon (file order) that contains ``(x, y)``."""
        for lane in self.lanes:
            if lane.contains(x, y):
                return lane.id
        return None

    def assign_zone(self, x: float, y: float) -> str | None:
        """First zone polygon (file order) that contains ``(x, y)``."""
        for zone in self.zones:
            if zone.contains(x, y):
                return zone.id
        return None

    def apply(self, trajectories: Sequence[Trajectory]) -> list[Trajectory]:
        """Return new trajectories with per-point ``lane`` and a majority track lane."""
        stamped: list[Trajectory] = []
        for traj in trajectories:
            points: list[TrajectoryPoint] = []
            votes: list[str] = []
            for point in traj.points:
                lane = self.assign_lane(point.center_x, point.center_y)
                points.append(replace(point, lane=lane))
                if lane is not None:
                    votes.append(lane)
            majority: str | None = None
            if votes:
                majority = max(set(votes), key=votes.count)
            stamped.append(
                Trajectory(
                    track_id=traj.track_id,
                    video_id=traj.video_id,
                    class_id=traj.class_id,
                    points=points,
                    lane=majority,
                )
            )
        return stamped
