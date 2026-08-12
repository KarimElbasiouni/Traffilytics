"""Traffilytics-owned video ingestion, frame extraction, and metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2

from computer_vision.preprocessing.config import REPO_ROOT

_SITE_PATTERNS = (
    re.compile(r"site[_\-]?(\d+)", re.IGNORECASE),
    re.compile(r"site[_\-]?([A-I])(?:[_\-]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])(\d{2})(?:[_\-]|$)", re.IGNORECASE),
    # DRIFT sample names like A_frame_0000 or A_od_small
    re.compile(r"(?:^|[_\-])([A-I])(?:[_\-]|$)", re.IGNORECASE),
)


class VideoProcessorError(Exception):
    """Raised when video load or processing fails."""


class VideoProcessor:
    """Load traffic video, extract frames, and persist FR-VID-003 metadata."""

    def __init__(
        self,
        video_path: str | Path,
        *,
        video_id: str | None = None,
        site: str | None = None,
        source: str = "drift",
        stabilized: bool | None = None,
        infer_stabilized_from_name: bool = True,
    ) -> None:
        """Set up paths and identity fields for one video before opening it.

        video_id defaults to the filename stem. site is inferred from the name
        when not provided (numeric or DRIFT letter A–I). stabilized can be set
        explicitly, or guessed from the filename when infer_stabilized_from_name
        is True.
        """
        self.video_path = Path(video_path).resolve()
        self._cap: cv2.VideoCapture | None = None
        self._metadata: dict[str, Any] | None = None

        stem = self.video_path.stem
        self.video_id = video_id or stem
        self.site = site if site is not None else self._infer_site(stem)
        self.source = source

        if stabilized is not None:
            self.stabilized = bool(stabilized)
        elif infer_stabilized_from_name:
            name_l = self.video_path.name.lower()
            self.stabilized = "stabil" in name_l
        else:
            self.stabilized = False

    @staticmethod
    def _infer_site(stem: str) -> str | None:
        """Guess the DRIFT site id from a filename stem, or return None."""
        for pattern in _SITE_PATTERNS:
            match = pattern.search(stem)
            if match:
                value = match.group(1)
                return value.upper() if value.isalpha() else value
        return None

    def load_video(self) -> cv2.VideoCapture:
        """Open and validate the video source."""
        if not self.video_path.is_file():
            raise VideoProcessorError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise VideoProcessorError(f"Unable to open video: {self.video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise VideoProcessorError(
                f"Invalid video properties (frames={frame_count}, "
                f"size={width}x{height}): {self.video_path}"
            )

        self._cap = cap
        self._metadata = None
        return cap

    def get_metadata(self) -> dict[str, Any]:
        """Return site, fps, resolution, duration, and related fields."""
        if self._metadata is not None:
            return dict(self._metadata)

        cap = self._cap or self.load_video()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0

        self._metadata = {
            "video_id": self.video_id,
            "site": self.site,
            "fps": round(fps, 4) if fps else 0.0,
            "resolution": f"{width}x{height}",
            "duration": round(duration, 3),
            "source": self.source,
            "stabilized": self.stabilized,
            "frame_count": frame_count,
            "path": str(self.video_path),
        }
        return dict(self._metadata)

    def extract_frames(
        self,
        out_dir: str | Path,
        *,
        stride: int = 1,
        max_frames: int | None = None,
        image_format: str = "jpg",
        jpeg_quality: int = 90,
    ) -> list[Path]:
        """Decode frames and write them under ``out_dir``.

        Args:
            out_dir: Directory for frame images.
            stride: Keep every Nth frame (NFR-PERF-002).
            max_frames: Optional cap on written frames.
            image_format: ``jpg`` or ``png``.
            jpeg_quality: JPEG quality 0–100.
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")

        cap = self._cap or self.load_video()
        # Rewind in case metadata already consumed the capture state
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fmt = image_format.lower().lstrip(".")
        if fmt not in {"jpg", "jpeg", "png"}:
            raise ValueError(f"Unsupported image_format: {image_format}")
        ext = "jpg" if fmt == "jpeg" else fmt
        encode_params: list[int] = []
        if ext == "jpg":
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        written: list[Path] = []
        frame_idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                filename = f"frame_{frame_idx:06d}.{ext}"
                dest = out_path / filename
                if not cv2.imwrite(str(dest), frame, encode_params):
                    raise VideoProcessorError(f"Failed to write frame: {dest}")
                written.append(dest)
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            frame_idx += 1

        return written

    def save_metadata(self, out_path: str | Path) -> Path:
        """Persist metadata JSON (FR-VID-003 shape plus helpful fields)."""
        meta = self.get_metadata()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
            fh.write("\n")
        return path

    def process(
        self,
        processed_root: str | Path,
        *,
        stride: int = 1,
        max_frames: int | None = None,
        image_format: str = "jpg",
        jpeg_quality: int = 90,
    ) -> dict[str, Any]:
        """Full ingest: frames + metadata under ``processed_root/<video_id>/``."""
        root = Path(processed_root)
        video_dir = root / self.video_id
        frames_dir = video_dir / "frames"

        self.load_video()
        meta = self.get_metadata()
        frame_paths = self.extract_frames(
            frames_dir,
            stride=stride,
            max_frames=max_frames,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
        meta_path = self.save_metadata(video_dir / "metadata.json")

        return {
            "metadata": meta,
            "metadata_path": str(meta_path),
            "frames_dir": str(frames_dir),
            "frames_written": len(frame_paths),
            "video_dir": str(video_dir),
        }

    def release(self) -> None:
        """Release the OpenCV capture if open."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoProcessor:
        """Open the video when used as a context manager (``with VideoProcessor(...)``)."""
        self.load_video()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Always release the capture when leaving the context manager."""
        self.release()


def default_processed_root() -> Path:
    """Return the default ``data/processed`` directory under the repo root."""
    return (REPO_ROOT / "data" / "processed").resolve()
