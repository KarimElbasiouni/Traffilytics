"""Qualitative tracking overlays: OBB + track_id on ingested frames (FR-TRK-005)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from computer_vision.detection.detector import (
    DetectorError,
    frame_index_from_path,
    list_frame_images,
)
from computer_vision.detection.types import Detection
from computer_vision.tracking.types import TrackedDetection

DEFAULT_OVERLAY_DIRNAME = "track_overlays"
DEFAULT_OVERLAY_VIDEO_NAME = "tracks_overlay.mp4"
_LINE_THICKNESS = 2
_TEXT_SCALE = 0.5
_TEXT_COLOR = (255, 255, 255)


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Stable BGR colour per track_id (not class — identity is the point)."""
    hue = int((track_id * 37) % 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracked_overlay(
    frame: np.ndarray,
    tracked: Sequence[TrackedDetection],
) -> np.ndarray:
    """Draw each tracked OBB and ``id:<track_id> <class>`` on a BGR copy."""
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.copy()
    for item in tracked:
        det: Detection = item.detection
        pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
        color = _color_for_track(item.track_id)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=_LINE_THICKNESS)
        xs = [p[0] for p in det.corners]
        ys = [p[1] for p in det.corners]
        origin = (int(min(xs)), max(12, int(min(ys)) - 4))
        label = f"id:{item.track_id} {det.class_name}"
        cv2.putText(
            canvas,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            _TEXT_SCALE,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _group_by_frame(
    tracked: Sequence[TrackedDetection],
) -> dict[int, list[TrackedDetection]]:
    grouped: dict[int, list[TrackedDetection]] = defaultdict(list)
    for item in tracked:
        grouped[item.frame].append(item)
    return grouped


def write_overlay_stills(
    frames_dir: str | Path,
    tracked: Sequence[TrackedDetection],
    dest_dir: str | Path,
    *,
    max_frames: int | None = None,
) -> list[Path]:
    """Write labelled overlay JPEGs under ``dest_dir`` from ingested frames."""
    images = list_frame_images(frames_dir)
    by_frame = _group_by_frame(tracked)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, image_path in enumerate(images):
        if max_frames is not None and i >= max_frames:
            break
        frame_idx = frame_index_from_path(image_path, fallback=i)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        canvas = draw_tracked_overlay(bgr, by_frame.get(frame_idx, []))
        out = dest / f"{image_path.stem}.jpg"
        if not cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise DetectorError(f"Failed to write overlay still: {out}")
        written.append(out)
    return written


def write_overlay_video(
    frames_dir: str | Path,
    tracked: Sequence[TrackedDetection],
    dest: str | Path,
    *,
    fps: float = 10.0,
) -> Path:
    """Encode a labelled overlay MP4 from ingested frames (OpenCV, no GPU)."""
    images = list_frame_images(frames_dir)
    if not images:
        raise DetectorError(f"No frames to overlay in {frames_dir}")
    by_frame = _group_by_frame(tracked)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise DetectorError(f"Could not read frame: {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(dest_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps) if fps > 0 else 10.0,
        (width, height),
    )
    if not writer.isOpened():
        raise DetectorError(f"Could not open overlay video writer: {dest_path}")
    try:
        for i, image_path in enumerate(images):
            frame_idx = frame_index_from_path(image_path, fallback=i)
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            canvas = draw_tracked_overlay(bgr, by_frame.get(frame_idx, []))
            if canvas.shape[0] != height or canvas.shape[1] != width:
                canvas = cv2.resize(canvas, (width, height))
            writer.write(canvas)
    finally:
        writer.release()
    return dest_path
