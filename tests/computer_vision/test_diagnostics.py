"""CPU tests for tracking diagnostics (no GT, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from computer_vision.detection.types import Detection
from computer_vision.tracking.diagnostics import TrackingDiagnostics, infer_frame_step
from computer_vision.tracking.types import TrackedDetection
from computer_vision.trajectories.generator import TrajectoryGenerator


def _tracked(frame: int, track_id: int, *, cx: float, cy: float = 100.0, class_id: int = 2) -> TrackedDetection:
    det = Detection.from_cxcywhr(
        frame=frame,
        class_id=class_id,
        confidence=0.9,
        center_x=cx,
        center_y=cy,
        width=40.0,
        height=20.0,
        angle=0.0,
    )
    return TrackedDetection(detection=det, track_id=track_id)


def test_infer_frame_step_is_gcd_of_gaps() -> None:
    tracked = [_tracked(0, 1, cx=10.0), _tracked(4, 1, cx=12.0), _tracked(8, 1, cx=14.0)]
    trajs = TrajectoryGenerator().generate(tracked, video_id="clip")
    assert infer_frame_step(trajs) == 4


def test_fragmentation_counts_intra_track_gaps() -> None:
    tracked = [_tracked(0, 1, cx=10.0), _tracked(1, 1, cx=12.0), _tracked(5, 1, cx=14.0)]
    trajs = TrajectoryGenerator().generate(tracked, video_id="clip")
    report = TrackingDiagnostics().report(trajs)
    assert report["frame_step"] == 1
    assert report["n_gaps"] == 1
    assert report["fragmented_tracks"] == 1


def test_suspected_id_switch_is_a_nearby_same_class_handoff() -> None:
    tracked = [
        _tracked(0, 1, cx=50.0),
        _tracked(1, 1, cx=55.0),
        _tracked(3, 2, cx=58.0),
        _tracked(4, 2, cx=62.0),
    ]
    trajs = TrajectoryGenerator().generate(tracked, video_id="clip")
    report = TrackingDiagnostics(id_switch_max_gap=5, id_switch_max_dist=80.0).report(trajs)
    assert report["suspected_id_switches"] == 1
    event = report["handoffs"][0]
    assert event["from_track_id"] == 1
    assert event["to_track_id"] == 2


def test_distant_restart_is_not_an_id_switch() -> None:
    tracked = [
        _tracked(0, 1, cx=10.0),
        _tracked(1, 1, cx=12.0),
        _tracked(3, 2, cx=400.0),
        _tracked(4, 2, cx=410.0),
    ]
    trajs = TrajectoryGenerator().generate(tracked, video_id="clip")
    report = TrackingDiagnostics(id_switch_max_dist=80.0).report(trajs)
    assert report["suspected_id_switches"] == 0


def test_write_diagnostics_json(tmp_path: Path) -> None:
    tracked = [_tracked(0, 1, cx=10.0), _tracked(1, 1, cx=12.0)]
    trajs = TrajectoryGenerator().generate(tracked, video_id="clip")
    dest = TrackingDiagnostics().write_json(trajs, tmp_path / "tracking_diagnostics.json")
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["n_tracks"] == 1
    assert "note" in payload
    assert "ground truth" in payload["note"].lower()
