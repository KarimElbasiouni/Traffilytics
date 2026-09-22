"""CPU tests for track_id overlay stills and video (no GPU)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from computer_vision.detection.types import Detection
from computer_vision.tracking.overlay import (
    draw_tracked_overlay,
    write_overlay_stills,
    write_overlay_video,
)
from computer_vision.tracking.types import TrackedDetection


def _write_jpeg(path: Path, color: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((48, 64, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return path


def _tracked(frame: int) -> TrackedDetection:
    det = Detection.from_cxcywhr(
        frame=frame,
        class_id=2,
        confidence=0.9,
        center_x=32.0,
        center_y=24.0,
        width=20.0,
        height=10.0,
        angle=0.0,
    )
    return TrackedDetection(detection=det, track_id=7)


def test_draw_tracked_overlay_keeps_shape() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    canvas = draw_tracked_overlay(frame, [_tracked(0)])
    assert canvas.shape == frame.shape
    assert canvas.dtype == np.uint8
    assert not np.array_equal(canvas, frame)


def test_write_overlay_stills_and_video(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _write_jpeg(frames / "frame_000000.jpg", 10)
    _write_jpeg(frames / "frame_000001.jpg", 20)
    tracked = [_tracked(0), _tracked(1)]
    stills = write_overlay_stills(frames, tracked, tmp_path / "track_overlays")
    assert len(stills) == 2
    assert stills[0].is_file()
    loaded = cv2.imread(str(stills[0]), cv2.IMREAD_COLOR)
    assert loaded is not None

    video = write_overlay_video(frames, tracked, tmp_path / "tracks_overlay.mp4", fps=5)
    assert video.is_file()
    assert video.stat().st_size > 0
