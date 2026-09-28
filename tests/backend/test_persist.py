"""Persist generated trajectories and analytics into SQLite (FR-DB-*)."""

from __future__ import annotations

from sqlalchemy import inspect, select

from backend.database.models import AnalyticsRow, EvaluationRun, Report, TrajectoryRow, Video
from backend.database.persist import get_trajectory_points, list_events, list_vehicles
from tests.backend.conftest import seed_completed_video


def test_schema_creates_tables(db) -> None:
    names = set(inspect(db.engine).get_table_names())
    for table in (
        "videos",
        "vehicles",
        "trajectories",
        "analytics",
        "events",
        "evaluation_runs",
        "jobs",
        "reports",
    ):
        assert table in names


def test_persist_writes_tracks_windows_events_and_report(db) -> None:
    analytics = seed_completed_video(db, "clip")
    with db.session_scope() as session:
        video = session.get(Video, "clip")
        assert video is not None
        assert video.status == "completed"
        assert video.site == "test_site"
        vehicles = list_vehicles(session, "clip")
        assert len(vehicles) == 2
        assert vehicles[0].vehicle_type == "car"
        points = get_trajectory_points(session, "clip", 1)
        assert len(points) == 4
        n_points = len(list(session.scalars(select(TrajectoryRow).where(TrajectoryRow.video_id == "clip"))))
        assert n_points == 8
        windows = list(session.scalars(select(AnalyticsRow).where(AnalyticsRow.video_id == "clip")))
        assert windows
        assert windows[0].units == "px"
        events = list_events(session, "clip")
        assert isinstance(events, list)
        report = session.get(Report, "clip")
        assert report is not None
        assert report.analytics_json["video_id"] == "clip"
        assert report.payload["title"].startswith("Transportation analysis")
        assert analytics["n_trajectories"] == 2
        runs = list(session.scalars(select(EvaluationRun).where(EvaluationRun.video_id == "clip")))
        assert runs
        assert runs[0].tracking_diagnostics["n_tracks"] == 2
