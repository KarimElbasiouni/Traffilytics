"""Shared fixtures for Epic 5 backend tests (CPU, no GPU, NFR-TEST-005)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from analytics.engine import AnalyticsEngine
from analytics.types import SceneContext
from backend.database.persist import persist_results, upsert_video
from backend.database.session import Database
from backend.services.settings import Settings, load_settings
from computer_vision.preprocessing.config import REPO_ROOT
from computer_vision.trajectories.types import Trajectory, TrajectoryPoint


def write_synthetic_mp4(
    path: Path, *, frames: int = 8, fps: float = 10.0, size: tuple[int, int] = (64, 48)
) -> Path:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "OpenCV VideoWriter failed to open"
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 20 % 255, 40, 80)
        writer.write(frame)
    writer.release()
    return path


def point(
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


def traj(
    track_id: int,
    coords: list[tuple[int, float, float]],
    *,
    lane: str | None = None,
    video_id: str = "clip",
) -> Trajectory:
    points = [point(track_id, f, x, y, lane=lane, video_id=video_id) for f, x, y in coords]
    return Trajectory(
        track_id=track_id,
        video_id=video_id,
        class_id=2,
        points=points,
        lane=lane,
    )


def sample_trajectories(video_id: str = "clip") -> list[Trajectory]:
    return [
        traj(1, [(0, 0.0, 10.0), (1, 10.0, 10.0), (2, 20.0, 10.0), (3, 30.0, 10.0)], video_id=video_id, lane="lane_1"),
        traj(2, [(0, 0.0, 40.0), (1, 8.0, 40.0), (2, 16.0, 40.0), (3, 24.0, 40.0)], video_id=video_id, lane="lane_2"),
    ]


def sample_analytics(video_id: str = "clip") -> dict:
    trajectories = sample_trajectories(video_id)
    context = SceneContext(
        video_id=video_id,
        fps=10.0,
        width=64,
        height=48,
        pixels_per_metre=None,
        assigner=None,
        source="test",
        tracker="bytetrack",
    )
    return AnalyticsEngine({"window_seconds": 1.0}).analyze(trajectories, context=context).to_dict()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    cfg = load_settings(
        repo_root=REPO_ROOT,
        db_url=f"sqlite:///{(tmp_path / 'test.db').resolve()}",
        data_root=data,
        config_path=REPO_ROOT / "configs" / "default.yaml",
        load_env=False,
    )
    cfg.lanes_dir = tmp_path / "lanes"
    cfg.lanes_dir.mkdir(parents=True, exist_ok=True)
    cfg.skip_detect = True
    cfg.skip_track = True
    return cfg


@pytest.fixture
def db(settings: Settings) -> Database:
    database = Database(settings.db_url)
    database.create_all()
    return database


def seed_completed_video(db: Database, video_id: str = "clip") -> dict:
    trajectories = sample_trajectories(video_id)
    analytics = sample_analytics(video_id)
    with db.session_scope(write=True) as session:
        upsert_video(
            session,
            video_id=video_id,
            site="test_site",
            duration=0.8,
            fps=10.0,
            resolution="64x48",
            source="upload",
            stabilized=False,
            status="processing",
            model_version="obb_v1",
            tracker_name="bytetrack",
        )
        persist_results(
            session,
            video_id=video_id,
            trajectories=trajectories,
            analytics=analytics,
            diagnostics={"n_tracks": len(trajectories), "suspected_id_switches": 0},
            model_version="obb_v1",
            tracker_name="bytetrack",
        )
    return analytics
