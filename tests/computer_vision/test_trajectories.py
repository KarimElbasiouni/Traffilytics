"""CPU tests for TrajectoryGenerator (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from computer_vision.detection.types import Detection
from computer_vision.tracking.types import TrackedDetection
from computer_vision.trajectories.generator import TrajectoryGenerator, summarize_tracks
from computer_vision.trajectories.types import TrajectoryPoint


def _tracked(frame: int, track_id: int, *, cx: float, class_id: int = 2) -> TrackedDetection:
    det = Detection.from_cxcywhr(
        frame=frame,
        class_id=class_id,
        confidence=0.91,
        center_x=cx,
        center_y=200.0,
        width=50.0,
        height=30.0,
        angle=0.25,
    )
    return TrackedDetection(detection=det, track_id=track_id)


def test_generate_groups_by_track_id_and_sorts_frames() -> None:
    tracked = [
        _tracked(2, 7, cx=30.0),
        _tracked(0, 3, cx=10.0),
        _tracked(1, 7, cx=20.0),
        _tracked(1, 3, cx=15.0),
    ]
    trajs = TrajectoryGenerator().generate(tracked, video_id="clip")
    assert [t.track_id for t in trajs] == [3, 7]
    assert [p.frame for p in trajs[0].points] == [0, 1]
    assert [p.frame for p in trajs[1].points] == [1, 2]
    assert trajs[0].entry_frame == 0
    assert trajs[0].exit_frame == 1
    assert trajs[0].lane is None
    assert trajs[0].class_name == "car"


def test_generate_drops_short_tracks() -> None:
    tracked = [_tracked(0, 1, cx=10.0), _tracked(0, 2, cx=80.0), _tracked(1, 2, cx=90.0)]
    trajs = TrajectoryGenerator(min_hits=2).generate(tracked, video_id="clip")
    assert [t.track_id for t in trajs] == [2]


def test_trajectory_point_matches_fr_trk_keys() -> None:
    point = TrajectoryPoint.from_tracked(_tracked(5, 52, cx=145.0), video_id="uav_clip_01")
    row = point.to_dict()
    assert row["track_id"] == 52
    assert row["frame"] == 5
    assert row["center_x"] == pytest.approx(145.0)
    assert row["class_id"] == 2
    assert row["video_id"] == "uav_clip_01"
    assert row["lane"] is None
    for key in (
        "track_id",
        "frame",
        "center_x",
        "center_y",
        "width",
        "height",
        "angle",
        "class_id",
        "confidence",
        "video_id",
        "lane",
    ):
        assert key in row


def test_write_json_includes_diagnostics(tmp_path: Path) -> None:
    tracked = [
        _tracked(0, 1, cx=10.0),
        _tracked(1, 1, cx=12.0),
        _tracked(0, 2, cx=80.0),
    ]
    gen = TrajectoryGenerator()
    trajs = gen.generate(tracked, video_id="clip")
    dest = gen.write_json(trajs, tmp_path / "trajectories.json", video_id="clip")
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["video_id"] == "clip"
    assert payload["tracker"] == "bytetrack"
    assert payload["n_tracks"] == 2
    assert payload["n_points"] == 3
    assert payload["diagnostics"]["singleton_tracks"] == 1
    assert payload["points"][0]["track_id"] == 1
    assert summarize_tracks(trajs)["n_tracks"] == 2
