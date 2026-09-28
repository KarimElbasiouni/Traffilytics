"""CPU tests for VehicleTracker and ByteTrack (no GPU, no video)."""

from __future__ import annotations

import pytest

from computer_vision.detection.types import Detection
from computer_vision.tracking.bytetrack import detections_to_results
from computer_vision.tracking.tracker import VehicleTracker, ingested_frame_order
from computer_vision.tracking.types import TrackedDetection


def _box(*, frame: int, cx: float, cy: float = 100.0, class_id: int = 2, conf: float = 0.9) -> Detection:
    return Detection.from_cxcywhr(
        frame=frame,
        class_id=class_id,
        confidence=conf,
        center_x=cx,
        center_y=cy,
        width=40.0,
        height=20.0,
        angle=0.0,
    )


class _StickyBackend:
    """Assign track_id 1 to every detection (deterministic, no Ultralytics)."""

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        return [TrackedDetection(detection=d, track_id=1) for d in detections]


def test_detections_to_results_empty() -> None:
    batch = detections_to_results([])
    assert len(batch) == 0
    assert batch.xyxy.shape == (0, 4)
    assert len(batch[batch.conf > 0.5]) == 0


def test_detections_to_results_aabb_and_xywhr() -> None:
    det = _box(frame=0, cx=50.0)
    batch = detections_to_results([det])
    assert len(batch) == 1
    assert batch.conf[0] == pytest.approx(0.9)
    assert batch.cls[0] == pytest.approx(2)
    x1, y1, x2, y2 = batch.xyxy[0]
    assert x1 < 50.0 < x2
    assert y1 < 100.0 < y2
    assert batch.xywhr[0, 0] == pytest.approx(50.0)
    assert batch.xywhr[0, 4] == pytest.approx(0.0)


def test_tracked_detection_rejects_negative_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        TrackedDetection(detection=_box(frame=0, cx=1.0), track_id=-1)


def test_vehicle_tracker_uses_injected_backend() -> None:
    dets = [_box(frame=i, cx=10.0 + i) for i in range(3)]
    tracker = VehicleTracker(backend=_StickyBackend())
    tracked = tracker.track(dets)
    assert len(tracked) == 3
    assert {t.track_id for t in tracked} == {1}
    assert [t.frame for t in tracked] == [0, 1, 2]
    assert tracked[0].detection.center_x == pytest.approx(10.0)


def test_ingested_frame_order_keeps_stride_without_empty_holes() -> None:
    assert ingested_frame_order({0: [], 2: [], 4: []}) == [0, 2, 4]
    assert ingested_frame_order({0: [], 1: [], 2: []}) == [0, 1, 2]
    assert ingested_frame_order({0: [], 1: [], 3: []}) == [0, 1, 2, 3]


def test_vehicle_tracker_skips_uningested_stride_frames() -> None:
    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[int | None] = []

        def update(self, detections: list[Detection]) -> list[TrackedDetection]:
            self.calls.append(detections[0].frame if detections else None)
            return [TrackedDetection(detection=d, track_id=1) for d in detections]

    dets = [_box(frame=i, cx=10.0) for i in (0, 2, 4)]
    backend = _Recorder()
    VehicleTracker(backend=backend).track(dets)
    assert backend.calls == [0, 2, 4]


def test_bytetrack_keeps_one_id_for_a_moving_vehicle() -> None:
    """A single car sliding right across frames should share one track_id."""
    pytest.importorskip("ultralytics")
    dets = [_box(frame=i, cx=80.0 + i * 12.0) for i in range(8)]
    tracker = VehicleTracker()
    tracked = tracker.track(dets)
    assert len(tracked) >= 3
    assert len({t.track_id for t in tracked}) == 1
    for item in tracked:
        assert item.detection.corners == dets[item.frame].corners


def test_bytetrack_separates_two_distant_vehicles() -> None:
    pytest.importorskip("ultralytics")
    dets: list[Detection] = []
    for i in range(8):
        dets.append(_box(frame=i, cx=60.0 + i * 8.0, cy=80.0, class_id=2))
        dets.append(_box(frame=i, cx=400.0 + i * 8.0, cy=300.0, class_id=5))
    tracker = VehicleTracker()
    tracked = tracker.track(dets)
    ids = {t.track_id for t in tracked}
    assert len(ids) == 2
    by_id: dict[int, set[int]] = {}
    for item in tracked:
        by_id.setdefault(item.track_id, set()).add(item.class_id)
    assert all(len(classes) == 1 for classes in by_id.values())
