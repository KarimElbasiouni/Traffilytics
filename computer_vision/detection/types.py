"""Detection record types and oriented-bounding-box geometry helpers.

FR-DET-002/003/004: class_id 0/1/2 (bus/car/truck), polygon OBBs, confidence
in [0, 1]. Geometry is unit-agnostic (pixels or normalized). Inference stores
pixel corners; adapter labels stay normalized.

Center/size/angle uses the Ultralytics xywhr convention: ``width`` lies along
``angle`` (radians), ``height`` is perpendicular. Trackers may consume either
corners or cxcywhr.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CLASS_NAMES: dict[int, str] = {0: "bus", 1: "car", 2: "truck"}

Point = tuple[float, float]
Corners = tuple[Point, Point, Point, Point]
# (center_x, center_y, width, height, angle_radians)
CxCyWhR = tuple[float, float, float, float, float]


def as_corners(values: Sequence[Any]) -> Corners:
    """Normalize nested pairs or a flat 8-tuple into four ``(x, y)`` corners.

    Accepts ``[[x, y], ...]`` (JSON / FR-DET) or ``(x1, y1, ..., x4, y4)``
    (YOLO-OBB / ``OBBBox.corners``).
    """
    if len(values) == 8:
        nums = [float(v) for v in values]
        return (
            (nums[0], nums[1]),
            (nums[2], nums[3]),
            (nums[4], nums[5]),
            (nums[6], nums[7]),
        )
    if len(values) == 4:
        pts: list[Point] = []
        for item in values:
            try:
                x, y = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Each corner must be an (x, y) pair, got {item!r}"
                ) from exc
            pts.append((float(x), float(y)))
        return (pts[0], pts[1], pts[2], pts[3])
    raise ValueError(
        f"Expected 4 corner pairs or 8 scalars, got {len(values)} values"
    )


def cxcywhr_to_corners(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle: float,
) -> Corners:
    """Convert center/size/angle (radians) to four ``(x, y)`` corners.

    Vertex order matches Ultralytics ``xywhr2xyxyxyxy``: ``center ± half-width
    along angle ± half-height perpendicular``.
    """
    if width < 0 or height < 0:
        raise ValueError(f"width and height must be non-negative, got {width}, {height}")
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    vx = (width / 2.0) * cos_a
    vy = (width / 2.0) * sin_a
    ux = (-height / 2.0) * sin_a
    uy = (height / 2.0) * cos_a
    return (
        (cx + vx + ux, cy + vy + uy),
        (cx + vx - ux, cy + vy - uy),
        (cx - vx - ux, cy - vy - uy),
        (cx - vx + ux, cy - vy + uy),
    )


def corners_to_cxcywhr(corners: Sequence[Any]) -> CxCyWhR:
    """Convert four polygon corners to ``(cx, cy, width, height, angle)``.

    Inverse of :func:`cxcywhr_to_corners` when vertices are in that same order
    (consecutive vertices around the rectangle). Angle is radians.
    """
    p1, p2, p3, p4 = as_corners(corners)
    cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
    cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0
    vx = (p1[0] + p2[0]) / 2.0 - cx
    vy = (p1[1] + p2[1]) / 2.0 - cy
    ux = (p1[0] + p4[0]) / 2.0 - cx
    uy = (p1[1] + p4[1]) / 2.0 - cy
    width = 2.0 * math.hypot(vx, vy)
    height = 2.0 * math.hypot(ux, uy)
    angle = math.atan2(vy, vx)
    return (cx, cy, width, height, angle)


@dataclass(frozen=True)
class Detection:
    """One vehicle detection: OBB polygon plus class and confidence (FR-DET).

    Corners are the canonical geometry. ``center_x`` / ``center_y`` / ``width`` /
    ``height`` / ``angle`` are derived so a tracker can consume either form.
    JSON uses the FR-DET key ``class`` (Python attribute is ``class_name``).
    """

    frame: int
    class_id: int
    confidence: float
    corners: Corners

    def __post_init__(self) -> None:
        """Coerce numeric fields and reject confidence outside ``[0, 1]``."""
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "corners", as_corners(self.corners))

    @property
    def class_name(self) -> str:
        """Human-readable class label (bus / car / truck), or the raw id."""
        return CLASS_NAMES.get(self.class_id, str(self.class_id))

    def as_cxcywhr(self) -> CxCyWhR:
        """Return ``(center_x, center_y, width, height, angle)`` in radians."""
        return corners_to_cxcywhr(self.corners)

    @property
    def center_x(self) -> float:
        """OBB center x (same units as corners)."""
        return self.as_cxcywhr()[0]

    @property
    def center_y(self) -> float:
        """OBB center y (same units as corners)."""
        return self.as_cxcywhr()[1]

    @property
    def width(self) -> float:
        """OBB extent along ``angle``."""
        return self.as_cxcywhr()[2]

    @property
    def height(self) -> float:
        """OBB extent perpendicular to ``angle``."""
        return self.as_cxcywhr()[3]

    @property
    def angle(self) -> float:
        """OBB rotation in radians (width direction)."""
        return self.as_cxcywhr()[4]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the FR-DET detection JSON object."""
        cx, cy, width, height, angle = self.as_cxcywhr()
        return {
            "frame": self.frame,
            "class_id": self.class_id,
            "class": self.class_name,
            "confidence": self.confidence,
            "center_x": cx,
            "center_y": cy,
            "width": width,
            "height": height,
            "angle": angle,
            "corners": [list(pt) for pt in self.corners],
        }

    @classmethod
    def from_cxcywhr(
        cls,
        *,
        frame: int,
        class_id: int,
        confidence: float,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        angle: float,
    ) -> Detection:
        """Build a detection from center/size/angle; corners are derived."""
        corners = cxcywhr_to_corners(center_x, center_y, width, height, angle)
        return cls(
            frame=frame,
            class_id=class_id,
            confidence=confidence,
            corners=corners,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Detection:
        """Build a detection from an FR-DET JSON object (corners or cxcywhr)."""
        if data.get("corners") is not None:
            return cls(
                frame=int(data["frame"]),
                class_id=int(data["class_id"]),
                confidence=float(data["confidence"]),
                corners=as_corners(data["corners"]),
            )
        return cls.from_cxcywhr(
            frame=int(data["frame"]),
            class_id=int(data["class_id"]),
            confidence=float(data["confidence"]),
            center_x=float(data["center_x"]),
            center_y=float(data["center_y"]),
            width=float(data["width"]),
            height=float(data["height"]),
            angle=float(data["angle"]),
        )
