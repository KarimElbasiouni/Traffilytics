"""Ultralytics BYTETracker adapter: OBB detections in, ``track_id``s out.

ByteTrack itself is not reimplemented (FR-TRK-002). Association uses the
detection's axis-aligned envelope plus ``xywhr`` so Ultralytics can keep the
oriented box; Traffilytics still stores the original OBB on the output.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from computer_vision.detection.types import Detection
from computer_vision.tracking.types import TrackedDetection

DEFAULT_TRACK_HIGH_THRESH = 0.25
DEFAULT_TRACK_LOW_THRESH = 0.1
DEFAULT_NEW_TRACK_THRESH = 0.25
DEFAULT_TRACK_BUFFER = 30
DEFAULT_MATCH_THRESH = 0.8


class TrackerError(Exception):
    """Raised when ByteTrack cannot be constructed or fed detections."""

    exit_code = 1


def _aabb_xyxy(det: Detection) -> tuple[float, float, float, float]:
    """Axis-aligned envelope of an OBB — ByteTrack's matching geometry."""
    xs = [p[0] for p in det.corners]
    ys = [p[1] for p in det.corners]
    return (min(xs), min(ys), max(xs), max(ys))


class _ByteTrackResults:
    """Minimal Results-like batch: ``conf``/``cls``/``xywh``/``xywhr``/``xyxy`` + mask indexing."""

    def __init__(
        self,
        xywh: np.ndarray,
        xywhr: np.ndarray,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float64)
        self.xywhr = np.asarray(xywhr, dtype=np.float64)
        self.xyxy = np.asarray(xyxy, dtype=np.float64)
        self.conf = np.asarray(conf, dtype=np.float64).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float64).reshape(-1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask: Any) -> _ByteTrackResults:
        return _ByteTrackResults(
            xywh=self.xywh[mask],
            xywhr=self.xywhr[mask],
            xyxy=self.xyxy[mask],
            conf=self.conf[mask],
            cls=self.cls[mask],
        )


def detections_to_results(detections: list[Detection]) -> _ByteTrackResults:
    """Pack a frame of :class:`Detection` into the object BYTETracker.update expects."""
    if not detections:
        return _ByteTrackResults(
            xywh=np.zeros((0, 4), dtype=np.float64),
            xywhr=np.zeros((0, 5), dtype=np.float64),
            xyxy=np.zeros((0, 4), dtype=np.float64),
            conf=np.zeros((0,), dtype=np.float64),
            cls=np.zeros((0,), dtype=np.float64),
        )
    xywh: list[list[float]] = []
    xywhr: list[list[float]] = []
    xyxy: list[list[float]] = []
    conf: list[float] = []
    cls: list[int] = []
    for det in detections:
        cx, cy, w, h, angle = det.as_cxcywhr()
        xywh.append([cx, cy, w, h])
        xywhr.append([cx, cy, w, h, angle])
        xyxy.append(list(_aabb_xyxy(det)))
        conf.append(det.confidence)
        cls.append(det.class_id)
    return _ByteTrackResults(
        xywh=np.asarray(xywh, dtype=np.float64),
        xywhr=np.asarray(xywhr, dtype=np.float64),
        xyxy=np.asarray(xyxy, dtype=np.float64),
        conf=np.asarray(conf, dtype=np.float64),
        cls=np.asarray(cls, dtype=np.float64),
    )


def _args_namespace(cfg: Mapping[str, Any] | None) -> SimpleNamespace:
    """Build BYTETracker args from a mapping, filling Ultralytics defaults."""
    data = dict(cfg or {})
    return SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=float(data.get("track_high_thresh", DEFAULT_TRACK_HIGH_THRESH)),
        track_low_thresh=float(data.get("track_low_thresh", DEFAULT_TRACK_LOW_THRESH)),
        new_track_thresh=float(data.get("new_track_thresh", DEFAULT_NEW_TRACK_THRESH)),
        track_buffer=int(data.get("track_buffer", DEFAULT_TRACK_BUFFER)),
        match_thresh=float(data.get("match_thresh", DEFAULT_MATCH_THRESH)),
        fuse_score=bool(data.get("fuse_score", True)),
    )


class ByteTrackBackend:
    """Thin wrapper around ``ultralytics.trackers.byte_tracker.BYTETracker``."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise TrackerError(
                "ultralytics is not installed. pip install ultralytics "
                "(or pip install 'traffilytics[ml]')."
            ) from exc
        self._engine = BYTETracker(_args_namespace(cfg))

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        """Associate this frame and return detections that received a ``track_id``."""
        output = self._engine.update(detections_to_results(detections), img=None)
        if output is None or len(output) == 0:
            return []
        rows = np.asarray(output, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        tracked: list[TrackedDetection] = []
        seen: set[int] = set()
        for row in rows:
            # [*coords, track_id, score, cls, idx]
            if row.size < 5:
                continue
            idx = int(row[-1])
            track_id = int(row[-4])
            if idx < 0 or idx >= len(detections) or idx in seen:
                continue
            seen.add(idx)
            tracked.append(TrackedDetection(detection=detections[idx], track_id=track_id))
        return tracked

    def reset(self) -> None:
        """Clear tracks and the global STrack id counter."""
        reset = getattr(self._engine, "reset", None)
        if callable(reset):
            reset()
