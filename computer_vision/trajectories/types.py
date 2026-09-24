"""Generated trajectory records (FR-TRK-003, FR-TRK-004)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from computer_vision.detection.types import CLASS_NAMES
from computer_vision.tracking.types import TrackedDetection


@dataclass(frozen=True)
class TrajectoryPoint:
    """One pose on a generated trajectory — the FR-TRK JSON object."""

    track_id: int
    frame: int
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float
    class_id: int
    confidence: float
    video_id: str
    lane: str | None = None
    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrajectoryPoint:
        """Rebuild a point from an FR-TRK JSON object."""
        raw_corners = data.get("corners")
        corners = None
        if raw_corners is not None:
            corners = tuple((float(pt[0]), float(pt[1])) for pt in raw_corners)
        class_id = data.get("class_id")
        if class_id is None:
            class_id = 0
        lane = data.get("lane")
        if lane is not None:
            lane = str(lane)
        return cls(
            track_id=int(data["track_id"]),
            frame=int(data["frame"]),
            center_x=float(data["center_x"]),
            center_y=float(data["center_y"]),
            width=float(data.get("width") or 0.0),
            height=float(data.get("height") or 0.0),
            angle=float(data.get("angle") or 0.0),
            class_id=int(class_id),
            confidence=float(data.get("confidence") or 0.0),
            video_id=str(data.get("video_id") or ""),
            lane=lane,
            corners=corners,
        )

    @classmethod
    def from_tracked(
        cls,
        tracked: TrackedDetection,
        *,
        video_id: str,
        lane: str | None = None,
    ) -> TrajectoryPoint:
        """Lift a tracked detection into a trajectory point (OBB pose preserved)."""
        det = tracked.detection
        cx, cy, width, height, angle = det.as_cxcywhr()
        return cls(
            track_id=tracked.track_id,
            frame=det.frame,
            center_x=cx,
            center_y=cy,
            width=width,
            height=height,
            angle=angle,
            class_id=det.class_id,
            confidence=det.confidence,
            video_id=video_id,
            lane=lane,
            corners=det.corners,
        )

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the FR-TRK trajectory-point object."""
        row: dict[str, Any] = {
            "track_id": self.track_id,
            "frame": self.frame,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "video_id": self.video_id,
            "lane": self.lane,
        }
        if self.corners is not None:
            row["corners"] = [list(pt) for pt in self.corners]
        return row


@dataclass
class Trajectory:
    """Time-ordered points for one ``track_id``."""

    track_id: int
    video_id: str
    class_id: int
    points: list[TrajectoryPoint] = field(default_factory=list)
    lane: str | None = None

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))

    @property
    def entry_frame(self) -> int:
        return self.points[0].frame if self.points else -1

    @property
    def exit_frame(self) -> int:
        return self.points[-1].frame if self.points else -1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trajectory:
        """Rebuild a track from ``trajectories.json`` ``tracks[]`` objects."""
        raw_points = data.get("points") or []
        points = [TrajectoryPoint.from_dict(row) for row in raw_points]
        points.sort(key=lambda p: p.frame)
        class_id = data.get("class_id")
        if class_id is None:
            class_id = points[0].class_id if points else 0
        video_id = str(data.get("video_id") or (points[0].video_id if points else ""))
        lane = data.get("lane")
        if lane is not None:
            lane = str(lane)
        return cls(
            track_id=int(data["track_id"]),
            video_id=video_id,
            class_id=int(class_id),
            points=points,
            lane=lane,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "video_id": self.video_id,
            "class_id": self.class_id,
            "class": self.class_name,
            "entry_frame": self.entry_frame,
            "exit_frame": self.exit_frame,
            "lane": self.lane,
            "n_points": len(self.points),
            "points": [p.to_dict() for p in self.points],
        }
