"""YOLO-OBB label records and parsing.

Label files carry one object per line in the Ultralytics OBB convention:
  class_id x1 y1 x2 y2 x3 y3 x4 y4
with normalized coordinates.

This is the generic YOLO-OBB label format rather than a property of any one
dataset, so it lives in the core detection package and dataset adapters build
on top of it. Keeping it here is what lets adapters depend on the core instead
of the reverse (NFR-MAINT-004).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from computer_vision.detection.types import CLASS_NAMES

# class_id plus four (x, y) corner pairs.
_OBB_FIELD_COUNT = 9


@dataclass(frozen=True)
class OBBBox:
    """One oriented bounding box: vehicle class plus eight normalized corner coords."""

    class_id: int
    corners: tuple[float, float, float, float, float, float, float, float]  # x1..y4

    @property
    def class_name(self) -> str:
        """Human-readable vehicle class label, or the raw id if unknown."""
        return CLASS_NAMES.get(self.class_id, str(self.class_id))


def parse_obb_label_line(line: str) -> OBBBox:
    """Parse one YOLO-OBB label line into an OBBBox (class + 8 corner values)."""
    parts = line.strip().split()
    if len(parts) != _OBB_FIELD_COUNT:
        raise ValueError(f"Expected 9 OBB fields (class + 8 coords), got {len(parts)}: {line!r}")
    class_id = int(float(parts[0]))
    coords = tuple(float(x) for x in parts[1:])
    return OBBBox(class_id=class_id, corners=coords)  # type: ignore[arg-type]


def load_obb_label_file(path: Path) -> list[OBBBox]:
    """Read every non-empty line of a ``.txt`` label file into a list of OBBBox objects."""
    if not path.is_file():
        raise FileNotFoundError(path)
    boxes: list[OBBBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        boxes.append(parse_obb_label_line(line))
    return boxes
