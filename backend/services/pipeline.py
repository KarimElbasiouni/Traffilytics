"""Worker: ingest → detect → track → analyze → persist (never inside an HTTP request)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from analytics.engine import AnalyticsEngine
from analytics.types import AnalyticsError, SceneContext
from backend.database.models import Job, Video
from backend.database.persist import persist_results, upsert_video
from backend.database.session import Database
from backend.services.jobs import update_job
from backend.services.lanes import lanes_from_video_json, load_lane_config, write_lane_config
from backend.services.settings import Settings
from computer_vision.preprocessing.video_processor import VideoProcessor, VideoProcessorError
from computer_vision.trajectories.generator import (
    DEFAULT_TRAJECTORIES_NAME,
    TrajectoryError,
    load_trajectories_json,
)
from computer_vision.trajectories.lanes import LaneAssigner, LaneConfigError

log = logging.getLogger(__name__)

DEFAULT_DETECTIONS_NAME = "detections.json"
DEFAULT_ANALYTICS_NAME = "analytics.json"
DEFAULT_DIAGNOSTICS_NAME = "tracking_diagnostics.json"


class PipelineError(Exception):
    """Raised when a pipeline stage cannot run."""


@dataclass
class ProcessOptions:
    model_version: str = "obb_v1"
    tracker: str = "bytetrack"
    scale_m_per_px: float | None = None
    lane_config: dict[str, Any] | None = None
    reuse_artifacts: bool = True
    skip_detect: bool = False
    skip_track: bool = False
    skip_analyze: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> ProcessOptions:
        data = dict(raw or {})
        scale = data.get("scale_m_per_px")
        return cls(
            model_version=str(data.get("model_version") or "obb_v1"),
            tracker=str(data.get("tracker") or "bytetrack"),
            scale_m_per_px=float(scale) if scale not in (None, "") else None,
            lane_config=data.get("lane_config") if isinstance(data.get("lane_config"), dict) else None,
            reuse_artifacts=bool(data.get("reuse_artifacts", True)),
            skip_detect=bool(data.get("skip_detect", False)),
            skip_track=bool(data.get("skip_track", False)),
            skip_analyze=bool(data.get("skip_analyze", False)),
        )

    def pixels_per_metre(self) -> float | None:
        if self.scale_m_per_px is None or self.scale_m_per_px <= 0:
            return None
        return 1.0 / float(self.scale_m_per_px)


class PipelineWorker:
    """Run one job id through the Traffilytics pipeline and persist results."""

    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def run_job(self, job_id: str) -> None:
        with self.db.session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                log.error("Job %s not found", job_id)
                return
            video_id = job.video_id
            options = ProcessOptions.from_mapping(job.options)
            video = session.get(Video, video_id)
            raw_path = None
            if video is not None:
                # Original file is stored as data/raw/<video_id>.*
                raw_path = _find_raw(self.settings.raw_dir, video_id)
        options.skip_detect = options.skip_detect or self.settings.skip_detect
        options.skip_track = options.skip_track or self.settings.skip_track
        options.reuse_artifacts = options.reuse_artifacts and self.settings.reuse_artifacts

        update_job(
            self.db,
            job_id,
            status="processing",
            stage="ingest",
            video_status="processing",
        )
        try:
            self._run_stages(job_id, video_id, raw_path, options)
            update_job(
                self.db,
                job_id,
                status="completed",
                stage="completed",
                progress=1.0,
                video_status="completed",
            )
        except (PipelineError, VideoProcessorError, AnalyticsError, TrajectoryError, LaneConfigError) as exc:
            log.warning("Job %s failed: %s", job_id, exc)
            update_job(
                self.db,
                job_id,
                status="failed",
                stage="failed",
                error=str(exc),
                video_status="failed",
            )
        except Exception as exc:
            log.exception("Job %s crashed", job_id)
            update_job(
                self.db,
                job_id,
                status="failed",
                stage="failed",
                error=str(exc),
                video_status="failed",
            )

    def _run_stages(
        self,
        job_id: str,
        video_id: str,
        raw_path: Path | None,
        options: ProcessOptions,
    ) -> None:
        processed = self.settings.processed_dir / video_id
        processed.mkdir(parents=True, exist_ok=True)

        update_job(self.db, job_id, stage="ingest")
        metadata = self._ingest(video_id, raw_path, processed)
        with self.db.session_scope(write=True) as session:
            upsert_video(
                session,
                video_id=video_id,
                site=metadata.get("site"),
                duration=metadata.get("duration"),
                fps=metadata.get("fps"),
                resolution=metadata.get("resolution"),
                source=metadata.get("source"),
                stabilized=metadata.get("stabilized"),
                status="processing",
                scale_m_per_px=options.scale_m_per_px,
                model_version=options.model_version,
                tracker_name=options.tracker,
            )

        assigner = self._resolve_assigner(video_id, options)

        detections_path = processed / DEFAULT_DETECTIONS_NAME
        trajectories_path = processed / DEFAULT_TRAJECTORIES_NAME
        analytics_path = processed / DEFAULT_ANALYTICS_NAME

        update_job(self.db, job_id, stage="detection")
        if not (options.reuse_artifacts and detections_path.is_file()):
            if options.skip_detect and not trajectories_path.is_file():
                raise PipelineError(
                    "Detection skipped and detections.json is missing. "
                    "Train weights live at models/your_obb.pt; run detect_frames.py "
                    "or disable skip_detect."
                )
            if not options.skip_detect:
                self._detect(video_id, processed, detections_path)

        update_job(self.db, job_id, stage="tracking")
        if not (options.reuse_artifacts and trajectories_path.is_file()):
            if options.skip_track:
                raise PipelineError("Tracking skipped and trajectories.json is missing.")
            self._track(video_id, detections_path, trajectories_path, assigner, options)

        trajectories, traj_meta = load_trajectories_json(trajectories_path)

        update_job(self.db, job_id, stage="analytics")
        self._write_overlay(processed, trajectories, metadata)
        if options.skip_analyze and analytics_path.is_file():
            analytics = json.loads(analytics_path.read_text(encoding="utf-8"))
        else:
            analytics = self._analyze(
                video_id,
                trajectories,
                metadata,
                assigner,
                options,
                tracker=str(traj_meta.get("tracker") or options.tracker),
            )
            analytics_path.write_text(json.dumps(analytics, indent=2) + "\n", encoding="utf-8")

        diagnostics = None
        diag_path = processed / DEFAULT_DIAGNOSTICS_NAME
        if diag_path.is_file():
            try:
                diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                diagnostics = None
        elif trajectories:
            from computer_vision.tracking.diagnostics import TrackingDiagnostics

            diagnostics = TrackingDiagnostics().report(trajectories)
            diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

        update_job(self.db, job_id, stage="persist")
        with self.db.session_scope(write=True) as session:
            persist_results(
                session,
                video_id=video_id,
                trajectories=trajectories,
                analytics=analytics if isinstance(analytics, dict) else {},
                diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
                model_version=options.model_version,
                tracker_name=options.tracker,
                scale_m_per_px=options.scale_m_per_px,
                lane_config=assigner.to_dict() if assigner is not None else None,
            )

    def _ingest(
        self,
        video_id: str,
        raw_path: Path | None,
        processed: Path,
    ) -> dict[str, Any]:
        meta_path = processed / "metadata.json"
        frames_dir = processed / "frames"
        if meta_path.is_file() and frames_dir.is_dir() and any(frames_dir.iterdir()):
            return json.loads(meta_path.read_text(encoding="utf-8"))
        if raw_path is None or not raw_path.is_file():
            if meta_path.is_file():
                return json.loads(meta_path.read_text(encoding="utf-8"))
            raise PipelineError(f"No uploaded video found for {video_id}")

        video_cfg = (self.settings.config.get("video") or {})
        stride = int(video_cfg.get("frame_stride") or 1)
        max_frames = video_cfg.get("max_frames")
        max_frames = int(max_frames) if max_frames is not None else None
        with VideoProcessor(raw_path, video_id=video_id, source="upload") as processor:
            result = processor.process(
                self.settings.processed_dir,
                stride=stride,
                max_frames=max_frames,
                image_format=str(video_cfg.get("frame_image_format") or "jpg"),
                jpeg_quality=int(video_cfg.get("jpeg_quality") or 90),
            )
        return dict(result["metadata"])

    def _detect(self, video_id: str, processed: Path, dest: Path) -> None:
        from computer_vision.detection.detector import (
            VehicleDetector,
            detect_and_write,
            list_frame_images,
        )

        det_cfg = self.settings.config.get("detection") or {}
        models_cfg = self.settings.config.get("models") or {}
        weights = Path(det_cfg.get("weights") or models_cfg.get("weights") or "models/your_obb.pt")
        if not weights.is_absolute():
            weights = (self.settings.repo_root / weights).resolve()
        frames = list_frame_images(processed / "frames")
        detector = VehicleDetector(
            weights,
            conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(det_cfg.get("iou_threshold", 0.7)),
            imgsz=int(det_cfg.get("imgsz", 640)),
            device=str(det_cfg.get("device") or "auto"),
            allow_pretrained=False,
        )
        detect_and_write(
            detector,
            frames,
            dest,
            video_id=video_id,
            weights=str(weights),
        )

    def _track(
        self,
        video_id: str,
        detections_path: Path,
        dest: Path,
        assigner: LaneAssigner | None,
        options: ProcessOptions,
    ) -> None:
        from computer_vision.detection.detector import load_detections_json
        from computer_vision.tracking.diagnostics import TrackingDiagnostics
        from computer_vision.tracking.tracker import VehicleTracker
        from computer_vision.trajectories.generator import TrajectoryGenerator

        if not detections_path.is_file():
            raise PipelineError(f"detections.json missing at {detections_path}")
        detections, _meta = load_detections_json(detections_path)
        track_cfg = self.settings.config.get("tracking") or {}
        tracker = VehicleTracker(track_cfg)
        tracked = tracker.track(detections)
        generator = TrajectoryGenerator(min_hits=int(track_cfg.get("min_hits") or 1))
        trajectories = generator.generate(tracked, video_id=video_id, assigner=assigner)
        generator.write_json(
            trajectories,
            dest,
            video_id=video_id,
            tracker=options.tracker,
        )
        diag = TrackingDiagnostics(
            id_switch_max_gap=int(track_cfg.get("id_switch_max_gap") or 5),
            id_switch_max_dist=float(track_cfg.get("id_switch_max_dist") or 80),
        ).report(trajectories)
        dest.with_name(DEFAULT_DIAGNOSTICS_NAME).write_text(
            json.dumps(diag, indent=2) + "\n", encoding="utf-8"
        )

    def _write_overlay(
        self,
        processed: Path,
        trajectories: list,
        metadata: Mapping[str, Any],
    ) -> None:
        """Best-effort boxed overlay MP4 for the dashboard (does not fail the job)."""
        frames_dir = processed / "frames"
        if not frames_dir.is_dir():
            return
        from computer_vision.detection.detector import (
            DEFAULT_DETECTIONS_NAME,
            DetectorError,
            list_frame_images,
            load_detections_json,
        )
        from computer_vision.tracking.overlay import (
            DEFAULT_OVERLAY_VIDEO_NAME,
            ensure_overlay_video,
            overlay_playback_fps,
        )

        detections = []
        det_path = processed / DEFAULT_DETECTIONS_NAME
        if det_path.is_file():
            try:
                detections, _meta = load_detections_json(det_path)
            except DetectorError as exc:
                log.warning("Overlay detections skipped for %s: %s", processed.name, exc)
        if not trajectories and not detections:
            return

        track_cfg = self.settings.config.get("tracking") or {}
        n_frames = len(list_frame_images(frames_dir))
        try:
            duration = float(metadata.get("duration") or 0) or None
        except (TypeError, ValueError):
            duration = None
        fallback = float(track_cfg.get("overlay_fps") or metadata.get("fps") or 10)
        fps = overlay_playback_fps(n_frames, duration, fallback)
        dest = processed / DEFAULT_OVERLAY_VIDEO_NAME
        try:
            ensure_overlay_video(
                frames_dir,
                trajectories,
                dest,
                detections=detections,
                fps=fps,
            )
        except (DetectorError, OSError) as exc:
            log.warning("Overlay video skipped for %s: %s", processed.name, exc)

    def _analyze(
        self,
        video_id: str,
        trajectories: list,
        metadata: Mapping[str, Any],
        assigner: LaneAssigner | None,
        options: ProcessOptions,
        *,
        tracker: str,
    ) -> dict[str, Any]:
        fps = float(metadata.get("fps") or 0.0)
        width, height = _parse_resolution(metadata.get("resolution"))
        an_cfg = dict(self.settings.config.get("analytics") or {})
        scale = options.pixels_per_metre()
        if scale is None:
            raw = an_cfg.get("pixels_per_metre")
            scale = float(raw) if raw not in (None, "") else None
        context = SceneContext(
            video_id=video_id,
            fps=fps,
            width=width,
            height=height,
            pixels_per_metre=scale,
            assigner=assigner,
            source="pipeline",
            tracker=tracker,
        )
        engine = AnalyticsEngine(an_cfg)
        report = engine.analyze(trajectories, context=context)
        return report.to_dict()

    def _resolve_assigner(self, video_id: str, options: ProcessOptions) -> LaneAssigner | None:
        if options.lane_config:
            assigner = LaneAssigner.from_mapping({**options.lane_config, "video_id": video_id})
            write_lane_config(assigner, video_id=video_id, lanes_dir=self.settings.lanes_dir)
            with self.db.session_scope(write=True) as session:
                video = session.get(Video, video_id)
                if video is not None:
                    video.lane_config = assigner.to_dict()
            return assigner
        with self.db.session_scope() as session:
            video = session.get(Video, video_id)
            stored = dict(video.lane_config) if video is not None and video.lane_config else None
        if stored:
            return lanes_from_video_json(stored, video_id=video_id)
        return load_lane_config(video_id, lanes_dir=self.settings.lanes_dir)


def _find_raw(raw_dir: Path, video_id: str) -> Path | None:
    if not raw_dir.is_dir():
        return None
    for path in sorted(raw_dir.iterdir()):
        if path.is_file() and path.stem == video_id:
            return path
    return None


def _parse_resolution(raw: Any) -> tuple[int | None, int | None]:
    if not raw:
        return None, None
    text = str(raw).lower().replace(" ", "")
    if "x" not in text:
        return None, None
    left, right = text.split("x", 1)
    try:
        return int(float(left)), int(float(right))
    except ValueError:
        return None, None
