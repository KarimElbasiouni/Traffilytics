"""CPU tests for LaneAssigner (FR-TRK-006)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from computer_vision.detection.types import Detection
from computer_vision.tracking.types import TrackedDetection
from computer_vision.trajectories.generator import TrajectoryGenerator
from computer_vision.trajectories.lanes import LaneAssigner, LaneConfigError


def _write_lanes(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "video_id": "clip",
                "lanes": [
                    {"id": "lane_1", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
                    {"id": "lane_2", "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]]},
                ],
                "zones": [
                    {"id": "zone_a", "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _tracked(frame: int, track_id: int, *, cx: float, cy: float = 50.0) -> TrackedDetection:
    det = Detection.from_cxcywhr(
        frame=frame,
        class_id=2,
        confidence=0.9,
        center_x=cx,
        center_y=cy,
        width=10.0,
        height=10.0,
        angle=0.0,
    )
    return TrackedDetection(detection=det, track_id=track_id)


def test_assign_lane_and_zone(tmp_path: Path) -> None:
    assigner = LaneAssigner.from_path(_write_lanes(tmp_path / "clip.json"))
    assert assigner.assign_lane(20.0, 20.0) == "lane_1"
    assert assigner.assign_lane(250.0, 20.0) == "lane_2"
    assert assigner.assign_lane(150.0, 20.0) is None
    assert assigner.assign_zone(10.0, 10.0) == "zone_a"
    assert assigner.assign_zone(80.0, 80.0) is None


def test_apply_stamps_points_and_majority_track_lane(tmp_path: Path) -> None:
    assigner = LaneAssigner.from_path(_write_lanes(tmp_path / "clip.json"))
    tracked = [
        _tracked(0, 1, cx=20.0),
        _tracked(1, 1, cx=30.0),
        _tracked(2, 1, cx=250.0),
        _tracked(0, 2, cx=150.0),
    ]
    trajs = TrajectoryGenerator().generate(tracked, video_id="clip", assigner=assigner)
    by_id = {t.track_id: t for t in trajs}
    assert [p.lane for p in by_id[1].points] == ["lane_1", "lane_1", "lane_2"]
    assert by_id[1].lane == "lane_1"
    assert by_id[2].points[0].lane is None
    assert by_id[2].lane is None


def test_from_path_rejects_bad_polygon(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"lanes": [{"id": "x", "polygon": [[0, 0]]}]}), encoding="utf-8")
    with pytest.raises(LaneConfigError, match="at least 3"):
        LaneAssigner.from_path(bad)


def test_discover_returns_none_when_missing(tmp_path: Path) -> None:
    assert LaneAssigner.discover("nope", lanes_dir=tmp_path) is None
    _write_lanes(tmp_path / "clip.json")
    found = LaneAssigner.discover("clip", lanes_dir=tmp_path)
    assert found is not None
    assert found.assign_lane(10, 10) == "lane_1"


def test_from_mapping_roundtrip() -> None:
    payload = {
        "video_id": "clip",
        "lanes": [{"id": "lane_1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
        "zones": [],
    }
    assigner = LaneAssigner.from_mapping(payload)
    assert assigner.to_dict()["lanes"][0]["id"] == "lane_1"
    assert assigner.assign_lane(2, 2) == "lane_1"
