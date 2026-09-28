"""Qualitative tracking overlays: OBB + track_id on ingested frames (FR-TRK-005)."""

from __future__ import annotations

import shutil
import subprocess
import sys
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
from computer_vision.detection.types import Detection, cxcywhr_to_corners
from computer_vision.tracking.types import TrackedDetection
from computer_vision.trajectories.types import Trajectory

DEFAULT_OVERLAY_DIRNAME = "track_overlays"
DEFAULT_OVERLAY_VIDEO_NAME = "tracks_overlay.mp4"
_LINE_THICKNESS = 2
_TEXT_SCALE = 0.5
_TEXT_COLOR = (255, 255, 255)
DASHBOARD_MAX_WIDTH = 1280


def overlay_playback_fps(
    n_frames: int,
    duration: float | None = None,
    fallback: float = 10.0,
) -> float:
    """Ingested-frame rate so overlay duration matches the source clip (stride-safe)."""
    if duration is not None and duration > 0 and n_frames > 1:
        return max(1.0, float(n_frames) / float(duration))
    return float(fallback) if fallback and fallback > 0 else 10.0


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Stable BGR colour per track_id (not class — identity is the point)."""
    hue = int((track_id * 37) % 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracked_overlay(
    frame: np.ndarray,
    tracked: Sequence[TrackedDetection],
    extra_detections: Sequence[Detection] = (),
) -> np.ndarray:
    """Draw tracked OBBs plus unmatched detections on a BGR copy."""
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.copy()
    for item in tracked:
        _paint_obb(
            canvas,
            item.detection,
            _color_for_track(item.track_id),
            f"id:{item.track_id} {item.detection.class_name}",
        )
    for det in extra_detections:
        _paint_obb(canvas, det, _color_for_class(det.class_id), det.class_name)
    return canvas


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    hue = int((int(class_id) * 28) % 180)
    hsv = np.uint8([[[hue, 180, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _paint_obb(
    canvas: np.ndarray,
    det: Detection,
    color: tuple[int, int, int],
    label: str,
) -> None:
    pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=_LINE_THICKNESS)
    xs = [p[0] for p in det.corners]
    ys = [p[1] for p in det.corners]
    origin = (int(min(xs)), max(12, int(min(ys)) - 4))
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


def _group_by_frame(
    tracked: Sequence[TrackedDetection],
) -> dict[int, list[TrackedDetection]]:
    grouped: dict[int, list[TrackedDetection]] = defaultdict(list)
    for item in tracked:
        grouped[item.frame].append(item)
    return grouped


def _group_detections(
    detections: Sequence[Detection],
) -> dict[int, list[Detection]]:
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        grouped[det.frame].append(det)
    return grouped


def _aabb(det: Detection) -> tuple[float, float, float, float]:
    xs = [p[0] for p in det.corners]
    ys = [p[1] for p in det.corners]
    return min(xs), min(ys), max(xs), max(ys)


def _aabb_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = _aabb(a)
    bx1, by1, bx2, by2 = _aabb(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def boxes_for_overlay(
    trajectories: Sequence[Trajectory],
    detections: Sequence[Detection] | None = None,
) -> tuple[list[TrackedDetection], list[Detection]]:
    """Prefer detector boxes; attach ``track_id`` when a trajectory overlaps."""
    tracked = tracked_from_trajectories(trajectories)
    if not detections:
        return tracked, []
    by_track = _group_by_frame(tracked)
    used: set[tuple[int, int]] = set()
    labelled: list[TrackedDetection] = []
    extra: list[Detection] = []
    for det in detections:
        best: TrackedDetection | None = None
        best_iou = 0.3
        for item in by_track.get(det.frame, []):
            key = (item.frame, item.track_id)
            if key in used:
                continue
            iou = _aabb_iou(det, item.detection)
            if iou > best_iou:
                best_iou = iou
                best = item
        if best is None:
            extra.append(det)
            continue
        used.add((best.frame, best.track_id))
        labelled.append(TrackedDetection(detection=det, track_id=best.track_id))
    return labelled, extra


def write_overlay_stills(
    frames_dir: str | Path,
    tracked: Sequence[TrackedDetection],
    dest_dir: str | Path,
    *,
    extra_detections: Sequence[Detection] = (),
    max_frames: int | None = None,
) -> list[Path]:
    """Write labelled overlay JPEGs under ``dest_dir`` from ingested frames."""
    images = list_frame_images(frames_dir)
    by_frame = _group_by_frame(tracked)
    extra = _group_detections(extra_detections)
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
        canvas = draw_tracked_overlay(
            bgr,
            by_frame.get(frame_idx, []),
            extra.get(frame_idx, []),
        )
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
    extra_detections: Sequence[Detection] = (),
    fps: float = 10.0,
    max_width: int | None = None,
) -> Path:
    """Encode a labelled overlay MP4 from ingested frames (OpenCV, no GPU)."""
    images = list_frame_images(frames_dir)
    if not images:
        raise DetectorError(f"No frames to overlay in {frames_dir}")
    by_frame = _group_by_frame(tracked)
    extra = _group_detections(extra_detections)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise DetectorError(f"Could not read frame: {images[0]}")
    src_h, src_w = first.shape[:2]
    width, height = src_w, src_h
    if max_width is not None and max_width > 0 and src_w > max_width:
        scale = max_width / float(src_w)
        width = int(max_width)
        height = max(1, int(round(src_h * scale)))
        if height % 2:
            height += 1
        if width % 2:
            width += 1
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
            canvas = draw_tracked_overlay(
                bgr,
                by_frame.get(frame_idx, []),
                extra.get(frame_idx, []),
            )
            if canvas.shape[1] != width or canvas.shape[0] != height:
                canvas = cv2.resize(canvas, (width, height))
            writer.write(canvas)
    finally:
        writer.release()
    return dest_path


def tracked_from_trajectories(trajectories: Sequence[Trajectory]) -> list[TrackedDetection]:
    """Rebuild tracked OBBs from generated trajectories so overlays do not need a re-track."""
    items: list[TrackedDetection] = []
    for traj in trajectories:
        for point in traj.points:
            corners = point.corners
            if corners is None:
                corners = cxcywhr_to_corners(
                    point.center_x,
                    point.center_y,
                    point.width,
                    point.height,
                    point.angle,
                )
            detection = Detection(
                frame=point.frame,
                class_id=point.class_id,
                confidence=point.confidence,
                corners=corners,
            )
            items.append(TrackedDetection(detection=detection, track_id=traj.track_id))
    return items


def _ffmpeg_bin() -> str | None:
    """Resolve ffmpeg even when it is on the conda base PATH, not the env PATH."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    here = Path(sys.executable).resolve().parent
    for parent in [here, *here.parents]:
        for cand in (parent / "ffmpeg", parent / "bin" / "ffmpeg"):
            if cand.is_file():
                return str(cand)
    return None


def _transcode_h264(src: Path) -> bool:
    """Replace ``src`` with an H.264 MP4 browsers can play. Returns False if ffmpeg is missing."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg or not src.is_file():
        return False
    tmp = src.with_name(src.stem + ".h264.mp4")
    for encoder in ("libx264", "libopenh264"):
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(src),
                    "-an",
                    "-c:v",
                    encoder,
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(tmp),
                ],
                capture_output=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            tmp.unlink(missing_ok=True)
            return False
        if result.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(src)
            return True
        tmp.unlink(missing_ok=True)
    return False


def ensure_overlay_video(
    frames_dir: str | Path,
    trajectories: Sequence[Trajectory],
    dest: str | Path,
    *,
    detections: Sequence[Detection] | None = None,
    fps: float = 10.0,
    max_width: int | None = DASHBOARD_MAX_WIDTH,
    force: bool = False,
) -> Path:
    """Write ``dest`` if missing: detector boxes on ingested frames, with track ids when known."""
    dest_path = Path(dest)
    if not force and dest_path.is_file() and dest_path.stat().st_size > 0:
        return dest_path
    tracked, extra = boxes_for_overlay(trajectories, detections)
    write_overlay_video(
        frames_dir,
        tracked,
        dest_path,
        extra_detections=extra,
        fps=fps,
        max_width=max_width,
    )
    _transcode_h264(dest_path)
    return dest_path
