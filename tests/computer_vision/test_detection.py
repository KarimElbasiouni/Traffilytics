"""Unit tests for Detection records and OBB geometry (no torch / GPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from computer_vision.detection.types import (
    CLASS_NAMES,
    Detection,
    as_corners,
    corners_to_cxcywhr,
    cxcywhr_to_corners,
)

FR_DET_KEYS = {
    "frame",
    "class_id",
    "class",
    "confidence",
    "center_x",
    "center_y",
    "width",
    "height",
    "angle",
    "corners",
}


def _sort_corners(corners: object) -> np.ndarray:
    """Order vertices by angle around the centroid for polygon comparison."""
    pts = np.asarray(as_corners(corners), dtype=np.float64)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def test_as_corners_nested_and_flat() -> None:
    """Nested pairs and a flat 8-tuple both become four (x, y) points."""
    nested = [[120.0, 200.0], [200.0, 200.0], [200.0, 260.0], [120.0, 260.0]]
    flat = (120.0, 200.0, 200.0, 200.0, 200.0, 260.0, 120.0, 260.0)
    assert as_corners(nested) == as_corners(flat)
    assert as_corners(nested)[0] == (120.0, 200.0)


def test_as_corners_rejects_bad_length() -> None:
    """Wrong number of values is a ValueError, not a silent reshape."""
    with pytest.raises(ValueError):
        as_corners((1.0, 2.0, 3.0))


@pytest.mark.parametrize(
    "pose",
    [
        (160.0, 230.0, 80.0, 60.0, 0.0),
        (160.0, 230.0, 80.0, 60.0, 0.42),
        (10.5, 20.25, 4.0, 2.0, math.pi / 2),
        (0.5, 0.5, 0.2, 0.1, math.pi / 6),
        (100.0, 50.0, 30.0, 30.0, -0.8),
    ],
)
def test_corners_cxcywhr_round_trip(
    pose: tuple[float, float, float, float, float],
) -> None:
    """cxcywhr → corners → cxcywhr recovers the original pose."""
    corners = cxcywhr_to_corners(*pose)
    recovered = corners_to_cxcywhr(corners)
    np.testing.assert_allclose(recovered, pose, atol=1e-9, rtol=0)


def test_corners_round_trip_preserves_polygon() -> None:
    """corners → cxcywhr → corners is the same rectangle (vertex order may wrap)."""
    original = cxcywhr_to_corners(160.0, 230.0, 80.0, 60.0, 0.42)
    recovered = cxcywhr_to_corners(*corners_to_cxcywhr(original))
    np.testing.assert_allclose(_sort_corners(original), _sort_corners(recovered), atol=1e-9)


def test_axis_aligned_synthetic_polygon() -> None:
    """An axis-aligned pixel box has the expected center; size follows edge 0."""
    corners = [[120, 200], [200, 200], [200, 260], [120, 260]]
    cx, cy, width, height, angle = corners_to_cxcywhr(corners)
    assert cx == pytest.approx(160.0)
    assert cy == pytest.approx(230.0)
    rebuilt = cxcywhr_to_corners(cx, cy, width, height, angle)
    np.testing.assert_allclose(_sort_corners(corners), _sort_corners(rebuilt), atol=1e-9)


def test_detection_json_matches_fr_det_fields() -> None:
    """to_dict keys and class/confidence match the FR-DET example object."""
    det = Detection.from_cxcywhr(
        frame=1204,
        class_id=1,
        confidence=0.94,
        center_x=160.0,
        center_y=230.0,
        width=80.0,
        height=60.0,
        angle=0.42,
    )
    payload = det.to_dict()
    assert set(payload) == FR_DET_KEYS
    assert payload["frame"] == 1204
    assert payload["class_id"] == 1
    assert payload["class"] == "car"
    assert payload["confidence"] == pytest.approx(0.94)
    assert payload["center_x"] == pytest.approx(160.0)
    assert payload["center_y"] == pytest.approx(230.0)
    assert payload["width"] == pytest.approx(80.0)
    assert payload["height"] == pytest.approx(60.0)
    assert payload["angle"] == pytest.approx(0.42)
    assert len(payload["corners"]) == 4
    assert all(len(pt) == 2 for pt in payload["corners"])


def test_detection_class_names() -> None:
    """class_id 0/1/2 map to bus/car/truck; unknown ids stringify."""
    corners = cxcywhr_to_corners(1.0, 1.0, 2.0, 1.0, 0.0)
    assert Detection(0, 0, 0.5, corners).class_name == "bus"
    assert Detection(0, 1, 0.5, corners).class_name == "car"
    assert Detection(0, 2, 0.5, corners).class_name == "truck"
    assert Detection(0, 9, 0.5, corners).class_name == "9"
    assert CLASS_NAMES == {0: "bus", 1: "car", 2: "truck"}


def test_detection_confidence_bounds() -> None:
    """Confidence outside [0, 1] is rejected (FR-DET-004)."""
    corners = cxcywhr_to_corners(1.0, 1.0, 2.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        Detection(frame=0, class_id=1, confidence=1.1, corners=corners)
    with pytest.raises(ValueError):
        Detection(frame=0, class_id=1, confidence=-0.01, corners=corners)


def test_detection_from_dict_round_trip() -> None:
    """FR-DET JSON round-trips through from_dict / to_dict."""
    det = Detection.from_cxcywhr(
        frame=12,
        class_id=0,
        confidence=1.0,
        center_x=10.0,
        center_y=20.0,
        width=8.0,
        height=4.0,
        angle=0.3,
    )
    restored = Detection.from_dict(det.to_dict())
    assert restored.frame == det.frame
    assert restored.class_id == det.class_id
    assert restored.class_name == "bus"
    np.testing.assert_allclose(restored.as_cxcywhr(), det.as_cxcywhr(), atol=1e-9)


def test_detection_from_dict_cxcywhr_only() -> None:
    """from_dict can rebuild corners when only center/size/angle are present."""
    data = {
        "frame": 3,
        "class_id": 2,
        "confidence": 0.5,
        "center_x": 5.0,
        "center_y": 6.0,
        "width": 10.0,
        "height": 4.0,
        "angle": 0.0,
    }
    det = Detection.from_dict(data)
    assert det.class_name == "truck"
    assert det.center_x == pytest.approx(5.0)
    assert det.width == pytest.approx(10.0)
    assert det.height == pytest.approx(4.0)
