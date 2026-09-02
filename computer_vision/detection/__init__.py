"""Detection package (Epic 2+): OBB types, training, evaluation, and inference."""

from computer_vision.detection.detector import DetectorError, VehicleDetector
from computer_vision.detection.types import (
    CLASS_NAMES,
    Detection,
    as_corners,
    corners_to_cxcywhr,
    cxcywhr_to_corners,
)

__all__ = [
    "CLASS_NAMES",
    "Detection",
    "DetectorError",
    "VehicleDetector",
    "as_corners",
    "corners_to_cxcywhr",
    "cxcywhr_to_corners",
]
