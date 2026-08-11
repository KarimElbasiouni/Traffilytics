"""Unit tests for VideoProcessor using a synthetic MP4 (no DRIFT required)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from computer_vision.preprocessing.video_processor import VideoProcessor, VideoProcessorError


def _write_synthetic_mp4(path: Path, *, frames: int = 12, fps: float = 10.0, size=(64, 48)) -> None:
    """Write a tiny colored MP4 so tests never need a real DRIFT video file."""
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "OpenCV VideoWriter failed to open"
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 20 % 255, 40, 80)
        cv2.putText(
            frame,
            str(i),
            (5, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Pytest fixture: temporary ``site_03_clip_01.mp4`` for metadata/site inference tests."""
    path = tmp_path / "site_03_clip_01.mp4"
    _write_synthetic_mp4(path)
    return path


def test_load_and_metadata(synthetic_video: Path) -> None:
    """Video opens and metadata includes id, site, fps, resolution, and duration."""
    with VideoProcessor(synthetic_video, source="test") as vp:
        meta = vp.get_metadata()

    assert meta["video_id"] == "site_03_clip_01"
    assert meta["site"] == "03"
    assert meta["source"] == "test"
    assert meta["resolution"] == "64x48"
    assert meta["fps"] > 0
    assert meta["frame_count"] >= 1
    assert meta["duration"] > 0
    assert meta["stabilized"] is False


def test_stabilized_inferred_from_name(tmp_path: Path) -> None:
    """Filenames containing ``stabil`` are marked stabilized=True automatically."""
    path = tmp_path / "site_01_stabilized.mp4"
    _write_synthetic_mp4(path, frames=4)
    with VideoProcessor(path) as vp:
        assert vp.get_metadata()["stabilized"] is True


def test_extract_frames_with_stride(synthetic_video: Path, tmp_path: Path) -> None:
    """Frame extraction with stride writes at least one JPEG under the output directory."""
    out_dir = tmp_path / "frames"
    with VideoProcessor(synthetic_video) as vp:
        written = vp.extract_frames(out_dir, stride=3, image_format="jpg")

    assert len(written) >= 1
    assert all(p.suffix == ".jpg" for p in written)
    assert all(p.is_file() for p in written)


def test_process_writes_metadata_and_frames(synthetic_video: Path, tmp_path: Path) -> None:
    """Full process() writes metadata.json and the requested number of frames."""
    processed_root = tmp_path / "processed"
    with VideoProcessor(synthetic_video, source="drift") as vp:
        result = vp.process(processed_root, stride=2, max_frames=3)

    meta_path = Path(result["metadata_path"])
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["video_id"] == "site_03_clip_01"
    assert meta["site"] == "03"
    assert result["frames_written"] == 3
    assert Path(result["frames_dir"]).is_dir()


def test_missing_video_raises(tmp_path: Path) -> None:
    """Loading a path that does not exist raises VideoProcessorError."""
    missing = tmp_path / "nope.mp4"
    with pytest.raises(VideoProcessorError):
        VideoProcessor(missing).load_video()
