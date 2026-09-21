"""Tracked detection records: a :class:`Detection` plus a ``track_id`` (FR-TRK-001)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from computer_vision.detection.types import CLASS_NAMES, Detection


@dataclass(frozen=True)
class TrackedDetection:
    """One detection after association: original OBB geometry plus ``track_id``.

    Geometry stays the detector's OBB. The tracker only assigns identity so
    Kalman AABB predictions never replace Traffilytics pose fields (FR-TRK-004).
    """

    detection: Detection
    track_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", int(self.track_id))
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")

    @property
    def frame(self) -> int:
        return self.detection.frame

    @property
    def class_id(self) -> int:
        return self.detection.class_id

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    def to_dict(self) -> dict[str, Any]:
        """FR-DET object plus ``track_id``."""
        row = self.detection.to_dict()
        row["track_id"] = self.track_id
        row["class"] = CLASS_NAMES.get(self.class_id, str(self.class_id))
        return row
