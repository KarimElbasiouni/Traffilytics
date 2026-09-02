"""Unit tests for Detection records, OBB geometry, and VehicleDetector (no GPU)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from computer_vision.detection.detector import (
    PRETRAINED_WEIGHTS,
    DetectorError,
    VehicleDetector,
)
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


# --- VehicleDetector (Ultralytics is stubbed; no weights / GPU required) ---


class _FakeOBB:
    """Minimal Ultralytics-style OBB container for stub models."""

    def __init__(
        self,
        *,
        xyxyxyxy: Any,
        conf: Any,
        cls: Any,
        xywhr: Any = None,
    ) -> None:
        self.xyxyxyxy = xyxyxyxy
        self.conf = conf
        self.cls = cls
        self.xywhr = xywhr


class _FakeResult:
    def __init__(self, obb: Any) -> None:
        self.obb = obb


class _StubYOLO:
    """Predictable YOLO stand-in: records predict kwargs and returns canned OBBs."""

    def __init__(self, obb: _FakeOBB | None) -> None:
        self.obb = obb
        self.last_source: Any = None
        self.last_kwargs: dict[str, Any] | None = None
        self.predict_calls = 0

    def predict(self, source: Any, **kwargs: Any) -> list[_FakeResult]:
        self.predict_calls += 1
        self.last_source = source
        self.last_kwargs = kwargs
        return [_FakeResult(self.obb)]


def _sample_corners() -> np.ndarray:
    """Three pixel rectangles, one per class (bus / car / truck)."""
    return np.array(
        [
            [[10.0, 10.0], [30.0, 10.0], [30.0, 24.0], [10.0, 24.0]],
            [[40.0, 40.0], [80.0, 40.0], [80.0, 70.0], [40.0, 70.0]],
            [[5.0, 80.0], [25.0, 80.0], [25.0, 100.0], [5.0, 100.0]],
        ],
        dtype=np.float64,
    )


def _stub_three_classes(*, extra_low_conf: bool = False) -> _StubYOLO:
    """Stub that emits class_id 0/1/2 with confidences in (threshold, 1]."""
    corners = _sample_corners()
    conf = [0.91, 0.84, 0.77]
    cls = [0, 1, 2]
    if extra_low_conf:
        corners = np.concatenate(
            [corners, [[[90.0, 90.0], [100.0, 90.0], [100.0, 96.0], [90.0, 96.0]]]],
            axis=0,
        )
        conf = conf + [0.10]
        cls = cls + [1]
    return _StubYOLO(
        _FakeOBB(xyxyxyxy=corners, conf=np.array(conf, dtype=np.float64), cls=np.array(cls))
    )


def test_vehicle_detector_stub_returns_class_ids_and_confidence() -> None:
    """Stub model yields class_id 0/1/2 and confidence in [0, 1] (FR-DET-002/004)."""
    stub = _stub_three_classes()
    detector = VehicleDetector(model=stub)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = detector.detect_objects(frame, frame_index=1204)

    assert [d.class_id for d in detections] == [0, 1, 2]
    assert [d.class_name for d in detections] == ["bus", "car", "truck"]
    assert all(0.0 <= d.confidence <= 1.0 for d in detections)
    assert all(d.frame == 1204 for d in detections)
    payload = detections[1].to_dict()
    assert payload["class"] == "car"
    assert payload["corners"][0] == [40.0, 40.0]


def test_vehicle_detector_applies_conf_threshold() -> None:
    """Boxes below conf_threshold are dropped even if the stub did not filter them."""
    stub = _stub_three_classes(extra_low_conf=True)
    detector = VehicleDetector(model=stub, conf_threshold=0.25)
    detections = detector.detect_objects(np.zeros((8, 8, 3), dtype=np.uint8))

    assert stub.last_kwargs is not None
    assert stub.last_kwargs["conf"] == pytest.approx(0.25)
    assert len(detections) == 3
    assert all(d.confidence >= 0.25 for d in detections)


def test_vehicle_detector_accepts_ndarray_and_path(tmp_path: Path) -> None:
    """detect_objects accepts a numpy frame or an existing image path."""
    stub = _stub_three_classes()
    detector = VehicleDetector(model=stub)
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    from_array = detector.detect_objects(array)
    assert len(from_array) == 3
    assert isinstance(stub.last_source, np.ndarray)

    image_path = tmp_path / "frame.jpg"
    # Avoid OpenCV in this unit: a tiny valid JPEG is not required; the detector
    # only checks that the path exists before handing it to the model.
    image_path.write_bytes(b"not-a-real-jpeg")
    from_path = detector.detect_objects(image_path, frame_index=7)
    assert len(from_path) == 3
    assert stub.last_source == str(image_path)
    assert from_path[0].frame == 7


def test_vehicle_detector_missing_image_path(tmp_path: Path) -> None:
    """A missing frame file fails with DetectorError (NFR-ACC-004)."""
    detector = VehicleDetector(model=_stub_three_classes())
    with pytest.raises(DetectorError, match="not found"):
        detector.detect_objects(tmp_path / "no_such_frame.jpg")


def test_vehicle_detector_missing_weights_error(tmp_path: Path) -> None:
    """Missing your_obb.pt fails clearly and does not load a pretrained fallback."""
    missing = tmp_path / "your_obb.pt"
    detector = VehicleDetector(weights=missing)
    with pytest.raises(DetectorError, match="OBB weights not found"):
        detector.load_model()
    with pytest.raises(DetectorError, match="FR-DET-001"):
        detector.detect_objects(np.zeros((4, 4, 3), dtype=np.uint8))


def test_vehicle_detector_allow_pretrained_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """allow_pretrained loads yolo11n-obb.pt with a loud FR-DET-001 warning."""
    seen: dict[str, str] = {}
    stub = _stub_three_classes()

    def fake_load(weights: str) -> _StubYOLO:
        seen["weights"] = weights
        return stub

    monkeypatch.setattr(
        "computer_vision.detection.detector._load_yolo", fake_load
    )
    detector = VehicleDetector(
        weights=tmp_path / "missing.pt",
        allow_pretrained=True,
    )
    with pytest.warns(UserWarning, match="FR-DET-001"):
        loaded = detector.load_model()
    assert loaded is stub
    assert seen["weights"] == PRETRAINED_WEIGHTS


def test_vehicle_detector_existing_weights_not_pretrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real weights file is used even when allow_pretrained is set."""
    weights = tmp_path / "your_obb.pt"
    weights.write_bytes(b"placeholder-weights")
    seen: dict[str, str] = {}

    def fake_load(name: str) -> _StubYOLO:
        seen["weights"] = name
        return _stub_three_classes()

    monkeypatch.setattr(
        "computer_vision.detection.detector._load_yolo", fake_load
    )
    detector = VehicleDetector(weights=weights, allow_pretrained=True)
    detector.load_model()
    assert seen["weights"] == str(weights.resolve())


def test_vehicle_detector_xywhr_fallback() -> None:
    """If xyxyxyxy is absent, corners are derived from xywhr (radians)."""
    cx, cy, width, height, angle = 50.0, 40.0, 20.0, 10.0, 0.0
    obb = _FakeOBB(
        xyxyxyxy=None,
        conf=[0.9],
        cls=[1],
        xywhr=np.array([[cx, cy, width, height, angle]], dtype=np.float64),
    )
    detector = VehicleDetector(model=_StubYOLO(obb))
    detections = detector.detect_objects(np.zeros((80, 80, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_id == 1
    np.testing.assert_allclose(
        detections[0].as_cxcywhr(),
        (cx, cy, width, height, angle),
        atol=1e-9,
    )


def test_vehicle_detector_empty_obb() -> None:
    """A frame with no OBB heads returns an empty list, not an error."""
    detector = VehicleDetector(model=_StubYOLO(None))
    assert detector.detect_objects(np.zeros((8, 8), dtype=np.uint8)) == []


def test_vehicle_detector_rejects_bad_conf_threshold() -> None:
    """Constructor rejects conf_threshold outside [0, 1]."""
    with pytest.raises(ValueError, match="conf_threshold"):
        VehicleDetector(model=_stub_three_classes(), conf_threshold=1.5)
