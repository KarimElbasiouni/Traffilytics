"""CPU tests for Epic 4 analytics (synthetic trajectories, no GPU, NFR-TEST-003)."""

from __future__ import annotations

from pathlib import Path

import pytest

from analytics.engine import AnalyticsEngine
from analytics.types import AnalyticsError, SceneContext
from computer_vision.trajectories.lanes import LaneAssigner, NamedPolygon
from computer_vision.trajectories.types import Trajectory, TrajectoryPoint


def _point(
    track_id: int,
    frame: int,
    x: float,
    y: float,
    *,
    lane: str | None = None,
    video_id: str = "clip",
) -> TrajectoryPoint:
    return TrajectoryPoint(
        track_id=track_id,
        frame=frame,
        center_x=x,
        center_y=y,
        width=10.0,
        height=10.0,
        angle=0.0,
        class_id=2,
        confidence=0.9,
        video_id=video_id,
        lane=lane,
    )


def _traj(
    track_id: int,
    coords: list[tuple[int, float, float]],
    *,
    lane: str | None = None,
) -> Trajectory:
    points = [_point(track_id, f, x, y, lane=lane) for f, x, y in coords]
    return Trajectory(
        track_id=track_id,
        video_id="clip",
        class_id=2,
        points=points,
        lane=lane,
    )


def _engine(**overrides: object) -> AnalyticsEngine:
    return AnalyticsEngine(overrides)


def _context(**overrides: object) -> SceneContext:
    data: dict = {
        "video_id": "clip",
        "fps": 10.0,
        "width": 400,
        "height": 200,
        "pixels_per_metre": None,
        "assigner": None,
        "source": "trajectories.json",
        "tracker": "bytetrack",
    }
    data.update(overrides)
    return SceneContext(**data)


def test_flow_speed_and_volume() -> None:
    # 10 px/frame at 10 fps → 100 px/s
    trajs = [
        _traj(1, [(0, 0.0, 0.0), (1, 10.0, 0.0), (2, 20.0, 0.0), (3, 30.0, 0.0)]),
        _traj(2, [(0, 0.0, 50.0), (1, 10.0, 50.0), (2, 20.0, 50.0), (3, 30.0, 50.0)]),
    ]
    report = _engine(window_seconds=1.0).analyze(trajs, context=_context())
    assert report.flow.mean_speed == pytest.approx(100.0)
    assert report.units.speed_unit == "px/s"
    assert report.units.labelled_as_physical is False
    assert "pixel-based" in report.units.to_dict()["note"]
    assert report.flow.vehicles_per_minute > 0
    assert report.flow.to_dict()["flow_density"]
    assert all("density" in row for row in report.flow.to_dict()["flow_density"])
    assert all(w.state in {"free", "congested", "queued", "unknown"} for w in report.flow.windows)


def test_flow_physical_units() -> None:
    trajs = [_traj(1, [(0, 0.0, 0.0), (10, 100.0, 0.0)])]
    report = _engine().analyze(trajs, context=_context(pixels_per_metre=10.0))
    # 100 px in 1 s → 100 px/s → 10 m/s
    assert report.units.speed_unit == "m/s"
    assert report.flow.mean_speed == pytest.approx(10.0)
    assert report.units.labelled_as_physical is True


def test_queued_state_when_stationary() -> None:
    trajs = [_traj(1, [(0, 5.0, 5.0), (5, 5.0, 5.0), (10, 5.0, 5.0), (15, 5.0, 5.0)])]
    report = _engine(window_seconds=2.0, stopped_speed_px_s=8.0).analyze(
        trajs, context=_context()
    )
    assert report.flow.windows
    assert all(w.state == "queued" for w in report.flow.windows)


def test_imbalance_skipped_without_lanes() -> None:
    trajs = [_traj(1, [(0, 0.0, 0.0), (1, 10.0, 0.0)])]
    report = _engine().analyze(trajs, context=_context())
    assert report.imbalance.configured is False
    assert report.imbalance.lanes == []


def test_imbalance_two_lanes() -> None:
    trajs = [
        _traj(1, [(0, 10.0, 10.0), (1, 20.0, 10.0)], lane="lane_1"),
        _traj(2, [(0, 12.0, 12.0), (1, 22.0, 12.0)], lane="lane_1"),
        _traj(3, [(0, 12.0, 12.0), (1, 22.0, 12.0)], lane="lane_1"),
        _traj(4, [(0, 210.0, 10.0), (1, 220.0, 10.0)], lane="lane_2"),
    ]
    assigner = LaneAssigner(
        video_id="clip",
        lanes=[
            NamedPolygon("lane_1", ((0, 0), (100, 0), (100, 50), (0, 50))),
            NamedPolygon("lane_2", ((200, 0), (300, 0), (300, 50), (200, 50))),
        ],
        zones=[],
    )
    report = _engine().analyze(trajs, context=_context(assigner=assigner))
    assert report.imbalance.configured is True
    by_lane = {row.lane: row for row in report.imbalance.lanes}
    assert by_lane["lane_1"].n_tracks == 3
    assert by_lane["lane_2"].n_tracks == 1
    assert report.imbalance.dominant_lane == "lane_1"
    assert report.imbalance.imbalance_index == pytest.approx(0.5)
    assert by_lane["lane_1"].mean_heading is not None


def test_bottleneck_skipped_without_zones() -> None:
    trajs = [_traj(1, [(0, 0.0, 0.0), (1, 10.0, 0.0)])]
    report = _engine().analyze(trajs, context=_context())
    assert report.bottleneck.configured is False
    assert report.bottleneck.primary is None
    assert report.bottleneck.heatmap  # occupancy grid still emitted


def test_bottleneck_slow_zone_is_primary() -> None:
    assigner = LaneAssigner(
        video_id="clip",
        lanes=[],
        zones=[
            NamedPolygon("zone_slow", ((0, 0), (80, 0), (80, 80), (0, 80))),
            NamedPolygon("zone_fast", ((200, 0), (280, 0), (280, 80), (200, 80))),
        ],
    )
    trajs = [
        _traj(1, [(0, 20.0, 20.0), (5, 21.0, 20.0), (10, 22.0, 20.0)]),
        _traj(2, [(0, 220.0, 20.0), (5, 270.0, 20.0), (10, 320.0, 20.0)]),
    ]
    report = _engine(window_seconds=2.0).analyze(trajs, context=_context(assigner=assigner))
    assert report.bottleneck.configured is True
    assert report.bottleneck.primary is not None
    assert report.bottleneck.primary["zone"] == "zone_slow"
    assert report.bottleneck.primary["cause"] in {
        "speed_reduction",
        "queue_formation",
        "accumulation",
    }


def test_stopped_vehicle_event() -> None:
    trajs = [_traj(9, [(0, 40.0, 40.0), (10, 40.0, 40.0), (20, 40.0, 40.0)])]
    report = _engine(stopped_min_seconds=1.0, stopped_speed_px_s=8.0).analyze(
        trajs, context=_context()
    )
    stopped = [e for e in report.events if e.type == "stopped_vehicle"]
    assert stopped
    assert stopped[0].track_id == 9
    assert stopped[0].time.count(":") == 2


def test_sudden_congestion_event() -> None:
    coords_fast = [(i, float(i * 40), 10.0) for i in range(0, 10)]
    coords_slow = [(i, 400.0, 10.0) for i in range(10, 20)]
    trajs = [_traj(1, coords_fast + coords_slow)]
    report = _engine(window_seconds=1.0, sudden_speed_drop=0.3).analyze(
        trajs, context=_context()
    )
    sudden = [e for e in report.events if e.type == "sudden_congestion"]
    assert sudden
    assert sudden[0].location == "scene"


def test_spillback_event_when_zone_queue_grows() -> None:
    assigner = LaneAssigner(
        video_id="clip",
        lanes=[],
        zones=[NamedPolygon("zone_east", ((0, 0), (100, 0), (100, 100), (0, 100)))],
    )
    moving = _traj(1, [(0, 20.0, 20.0), (2, 50.0, 20.0), (4, 80.0, 20.0)])
    queued = [
        _traj(n, [(10, 20.0 + n, 20.0), (12, 20.0 + n, 20.0), (14, 20.0 + n, 20.0)])
        for n in range(2, 6)
    ]
    report = _engine(
        window_seconds=1.0,
        stopped_speed_px_s=8.0,
        queue_fraction=0.4,
    ).analyze([moving, *queued], context=_context(assigner=assigner))
    spill = [e for e in report.events if e.type == "queue_spillback"]
    assert spill
    assert spill[0].location == "zone_east"


def test_insights_cover_bottleneck_and_imbalance() -> None:
    assigner = LaneAssigner(
        video_id="clip",
        lanes=[
            NamedPolygon("lane_1", ((0, 0), (100, 0), (100, 50), (0, 50))),
            NamedPolygon("lane_2", ((200, 0), (300, 0), (300, 50), (200, 50))),
        ],
        zones=[NamedPolygon("zone_a", ((0, 0), (80, 0), (80, 80), (0, 80)))],
    )
    trajs = [
        _traj(1, [(0, 10.0, 10.0), (1, 11.0, 10.0)], lane="lane_1"),
        _traj(2, [(0, 210.0, 10.0), (1, 250.0, 10.0)], lane="lane_2"),
    ]
    report = _engine().analyze(trajs, context=_context(assigner=assigner))
    by_id = {i.id: i.text for i in report.insights}
    assert "bottleneck" in by_id["bottleneck"].lower()
    assert "imbalance" in by_id["imbalance"].lower() or "lane" in by_id["imbalance"].lower()
    assert "stopped" in by_id["events"]


def test_empty_trajectories() -> None:
    report = _engine().analyze([], context=_context())
    assert report.n_trajectories == 0
    assert report.insights[0].id == "empty"


def test_rejects_non_positive_fps() -> None:
    with pytest.raises(AnalyticsError, match="fps"):
        _engine().analyze([_traj(1, [(0, 0.0, 0.0), (1, 1.0, 0.0)])], context=_context(fps=0.0))


def test_write_json_roundtrip(tmp_path: Path) -> None:
    trajs = [_traj(1, [(0, 0.0, 0.0), (1, 10.0, 0.0)])]
    engine = _engine()
    report = engine.analyze(trajs, context=_context())
    dest = engine.write_json(report, tmp_path / "analytics.json")
    payload = dest.read_text(encoding="utf-8")
    assert "flow_density" in payload
    assert "pixel-based" in payload
    assert report.to_dict()["n_events"] == len(report.events)
