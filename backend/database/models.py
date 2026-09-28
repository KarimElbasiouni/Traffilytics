"""SQLAlchemy models for videos, generated tracks, analytics, jobs, and reports.

Tables follow ``docs/08_Database_Design.md`` (FR-DB-001 … FR-DB-006) plus a
``jobs`` table for FR-API-003 progress and a ``reports`` row for FR-RPT-001.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now_iso() -> str:
    """UTC timestamp as ISO-8601 text (SQLite-friendly)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Base(DeclarativeBase):
    """Declarative base for Traffilytics tables."""


class Video(Base):
    """Clip metadata and processing status (FR-DB-001)."""

    __tablename__ = "videos"
    __table_args__ = (Index("ix_videos_model_version", "model_version"),)

    video_id: Mapped[str] = mapped_column(String, primary_key=True)
    site: Mapped[str | None] = mapped_column(String, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    stabilized: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String, default="queued")
    scale_m_per_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    lane_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    tracker_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=utc_now_iso)

    vehicles: Mapped[list[Vehicle]] = relationship(back_populates="video")
    jobs: Mapped[list[Job]] = relationship(back_populates="video")


class Vehicle(Base):
    """One generated track (FR-DB-002)."""

    __tablename__ = "vehicles"

    video_id: Mapped[str] = mapped_column(
        String, ForeignKey("videos.video_id"), primary_key=True
    )
    track_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_id: Mapped[int] = mapped_column(Integer)
    vehicle_type: Mapped[str | None] = mapped_column(String, nullable=True)
    entry_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entry_time: Mapped[str | None] = mapped_column(String, nullable=True)
    exit_time: Mapped[str | None] = mapped_column(String, nullable=True)

    video: Mapped[Video] = relationship(back_populates="vehicles")
    points: Mapped[list[TrajectoryRow]] = relationship(back_populates="vehicle")


class TrajectoryRow(Base):
    """One generated pose (FR-DB-003)."""

    __tablename__ = "trajectories"
    __table_args__ = (
        ForeignKeyConstraint(
            ["video_id", "track_id"],
            ["vehicles.video_id", "vehicles.track_id"],
        ),
        Index("ix_trajectories_video_frame", "video_id", "frame"),
        Index("ix_trajectories_video_lane", "video_id", "lane"),
    )

    video_id: Mapped[str] = mapped_column(String, primary_key=True)
    track_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    frame: Mapped[int] = mapped_column(Integer, primary_key=True)
    center_x: Mapped[float] = mapped_column(Float)
    center_y: Mapped[float] = mapped_column(Float)
    width: Mapped[float | None] = mapped_column(Float, nullable=True)
    height: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    class_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lane: Mapped[str | None] = mapped_column(String, nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)
    speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    acceleration: Mapped[float | None] = mapped_column(Float, nullable=True)
    heading: Mapped[float | None] = mapped_column(Float, nullable=True)

    vehicle: Mapped[Vehicle] = relationship(back_populates="points")


class AnalyticsRow(Base):
    """Time-window flow metrics (FR-DB-004)."""

    __tablename__ = "analytics"
    __table_args__ = (Index("ix_analytics_video", "video_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.video_id"))
    frame_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vehicle_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    average_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    density: Mapped[float | None] = mapped_column(Float, nullable=True)
    units: Mapped[str | None] = mapped_column(String, nullable=True)
    traffic_state: Mapped[str | None] = mapped_column(String, nullable=True)
    congestion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)


class Event(Base):
    """Rule-based traffic event (FR-DB-005)."""

    __tablename__ = "events"
    __table_args__ = (Index("ix_events_video", "video_id"),)

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.video_id"))
    event_type: Mapped[str] = mapped_column(String)
    frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[str | None] = mapped_column(String, nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class EvaluationRun(Base):
    """Detector metrics / tracking diagnostics, separate from live tracks (FR-DB-006)."""

    __tablename__ = "evaluation_runs"

    eval_id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("videos.video_id"), nullable=True
    )
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    tracker_name: Mapped[str | None] = mapped_column(String, nullable=True)
    detection_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tracking_diagnostics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=utc_now_iso)


class Job(Base):
    """Async upload/process job (FR-API-003, NFR-PERF-005)."""

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.video_id"))
    kind: Mapped[str] = mapped_column(String, default="process")
    status: Mapped[str] = mapped_column(String, default="queued")
    stage: Mapped[str] = mapped_column(String, default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    options: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=utc_now_iso)
    updated_at: Mapped[str] = mapped_column(String, default=utc_now_iso)

    video: Mapped[Video] = relationship(back_populates="jobs")


class Report(Base):
    """Automated transportation report plus the full analytics JSON (FR-RPT-001)."""

    __tablename__ = "reports"

    video_id: Mapped[str] = mapped_column(
        String, ForeignKey("videos.video_id"), primary_key=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    insights: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    analytics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    generated_at: Mapped[str] = mapped_column(String, default=utc_now_iso)
