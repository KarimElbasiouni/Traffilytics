"""Generated trajectory records (FR-TRK-003, FR-TRK-004)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
