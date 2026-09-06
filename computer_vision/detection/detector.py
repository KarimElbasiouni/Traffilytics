"""Traffilytics-trained YOLO OBB inference (FR-DET-001 … FR-DET-004).

``VehicleDetector`` wraps Ultralytics ``YOLO`` and returns :class:`Detection`
records with **pixel** corners. Product weights are ``models/your_obb.pt``
(NFR-ACC-004: missing weights fail with a clear error). ``allow_pretrained``
is a CPU/wiring escape hatch that loads ``yolo11n-obb.pt`` with a warning; it
does **not** satisfy FR-DET-001.

Batch helpers write ``data/processed/<video_id>/detections.json`` (Epic 3 input)
and optional ``det_overlays/``.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from computer_vision.detection.types import (
    Detection,
    as_corners,
    cxcywhr_to_corners,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = _REPO_ROOT / "models" / "your_obb.pt"
PRETRAINED_WEIGHTS = "yolo11n-obb.pt"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_IMGSZ = 640
DEFAULT_DETECTIONS_NAME = "detections.json"
DEFAULT_OVERLAY_DIRNAME = "det_overlays"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FRAME_INDEX_RE = re.compile(r"(\d+)$")

_PRETRAINED_WARNING = (
    "VehicleDetector is using pretrained yolo11n-obb.pt, not models/your_obb.pt. "
    "This is CPU/wiring only and does not satisfy FR-DET-001."
)


class DetectorError(Exception):
    """Raised when OBB weights cannot be loaded or a frame cannot be inferred (NFR-ACC-004)."""

    exit_code = 1


def _load_yolo(weights: str) -> Any:
    """Import Ultralytics and construct ``YOLO(weights)``.

    Isolated so tests can monkeypatch without importing torch.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError(
            "ultralytics is not installed. pip install ultralytics "
            "(or pip install 'traffilytics[ml]')."
        ) from exc
    return YOLO(weights)


def _resolve_device(device: str) -> str:
    """Map ``auto`` to ``cuda`` when a GPU is present, otherwise ``cpu``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _to_1d_array(value: Any) -> np.ndarray:
    """Convert a tensor / list / scalar to a 1-D numpy array (empty if missing)."""
    if value is None:
        return np.zeros((0,), dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.atleast_1d(np.asarray(value, dtype=np.float64)).reshape(-1)


def _corners_from_row(row: Any) -> tuple[tuple[float, float], ...]:
    """Turn one OBB row (8 scalars or 4×2) into four ``(x, y)`` pixel corners."""
    flat = np.asarray(row, dtype=np.float64).reshape(-1)
    if flat.size != 8:
        raise DetectorError(f"Expected 8 corner coordinates, got {flat.size}")
    return as_corners(flat.tolist())


def _as_predict_source(frame: np.ndarray | str | Path) -> np.ndarray | str:
    """Accept an image array or filesystem path; reject missing files (NFR-ACC-004)."""
    if isinstance(frame, (str, Path)):
        path = Path(frame)
        if not path.is_file():
            raise DetectorError(f"Frame image not found: {path}")
        return str(path)
    if isinstance(frame, np.ndarray):
        if frame.ndim not in (2, 3):
            raise DetectorError(
                f"Frame array must be H×W or H×W×C, got shape {frame.shape}"
            )
        return frame
    raise TypeError(
        f"frame must be a numpy ndarray or image path, got {type(frame).__name__}"
    )


class VehicleDetector:
    """Load Traffilytics OBB weights and detect vehicles on a single frame.

    By default only ``models/your_obb.pt`` is loaded. Inject ``model`` to stub
    Ultralytics in tests. ``detect_objects`` accepts a numpy frame or image path.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = "auto",
        allow_pretrained: bool = False,
        model: Any | None = None,
    ) -> None:
        """Configure inference; weights are loaded lazily by :meth:`load_model`.

        ``conf_threshold`` is applied both as the Ultralytics ``conf=`` argument
        and as a post-filter so stub models cannot leak low-score boxes.
        """
        if not 0.0 <= float(conf_threshold) <= 1.0:
            raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold}")
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if int(imgsz) <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}")

        self.weights = Path(weights) if weights is not None else DEFAULT_WEIGHTS
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.allow_pretrained = bool(allow_pretrained)
        self._model = model

    def load_model(self) -> Any:
        """Load Traffilytics OBB weights (or the injected stub).

        Raises:
            DetectorError: weights file missing and ``allow_pretrained`` is false,
                or Ultralytics is not installed.
        """
        if self._model is not None:
            return self._model

        weights_arg = self._resolve_weights_arg()
        self._model = _load_yolo(weights_arg)
        return self._model

    def detect_objects(
        self,
        frame: np.ndarray | str | Path,
        *,
        frame_index: int = 0,
    ) -> list[Detection]:
        """Run OBB inference on one frame and return FR-DET :class:`Detection` rows.

        ``frame`` is a BGR/RGB ``ndarray`` or a path to an image. Corners are in
        **pixels** of that frame. ``frame_index`` is stored on each record for
        later ``detections.json`` export.
        """
        source = _as_predict_source(frame)
        model = self.load_model()
        raw = self._predict(model, source)
        return self._parse_results(raw, frame_index=int(frame_index))

    def _resolve_weights_arg(self) -> str:
        """Return a local ``.pt`` path, or the pretrained name when allowed."""
        path = self.weights
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        else:
            path = path.resolve()

        if path.is_file():
            return str(path)

        if self.allow_pretrained:
            warnings.warn(_PRETRAINED_WARNING, UserWarning, stacklevel=3)
            return PRETRAINED_WEIGHTS

        raise DetectorError(
            f"Traffilytics OBB weights not found: {path}. "
            "Train on a CUDA host with: python scripts/train_obb.py "
            "(copies best.pt to models/your_obb.pt). "
            "For CPU wiring only, pass allow_pretrained=True to use "
            "yolo11n-obb.pt — that is not FR-DET-001 compliant."
        )

    def _predict(self, model: Any, source: np.ndarray | str) -> Any:
        """Call Ultralytics ``predict`` (or the model itself) with detector settings."""
        kwargs: dict[str, Any] = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.imgsz,
            "device": _resolve_device(self.device),
            "verbose": False,
        }
        predict = getattr(model, "predict", None)
        if callable(predict):
            return predict(source, **kwargs)
        return model(source, **kwargs)

    def _parse_results(self, raw: Any, *, frame_index: int) -> list[Detection]:
        """Convert Ultralytics OBB results (or stub equivalents) into detections."""
        if raw is None:
            return []
        results = raw if isinstance(raw, (list, tuple)) else [raw]
        detections: list[Detection] = []
        for result in results:
            detections.extend(self._detections_from_obb(result, frame_index=frame_index))
        return detections

    def _detections_from_obb(self, result: Any, *, frame_index: int) -> list[Detection]:
        """Parse ``result.obb`` (xyxyxyxy preferred, xywhr fallback)."""
        obb = getattr(result, "obb", None)
        if obb is None:
            return []

        confs = _to_1d_array(getattr(obb, "conf", None))
        classes = _to_1d_array(getattr(obb, "cls", None))
        corners_rows = self._obb_corner_rows(obb, expected=len(confs))
        count = min(len(confs), len(classes), len(corners_rows))

        detections: list[Detection] = []
        for i in range(count):
            conf = float(confs[i])
            if conf < self.conf_threshold or not 0.0 <= conf <= 1.0:
                continue
            detections.append(
                Detection(
                    frame=frame_index,
                    class_id=int(classes[i]),
                    confidence=conf,
                    corners=_corners_from_row(corners_rows[i]),
                )
            )
        return detections

    def _obb_corner_rows(self, obb: Any, *, expected: int) -> list[Any]:
        """Prefer polygon corners; fall back to center/size/angle (radians)."""
        xyxyxyxy = getattr(obb, "xyxyxyxy", None)
        if xyxyxyxy is not None:
            arr = np.asarray(
                _tensor_to_numpy(xyxyxyxy),
                dtype=np.float64,
            )
            if arr.size == 0:
                return []
            return list(arr.reshape(-1, 8))

        xywhr = getattr(obb, "xywhr", None)
        if xywhr is None:
            return []
        arr = np.asarray(_tensor_to_numpy(xywhr), dtype=np.float64)
        if arr.size == 0:
            return []
        rows = arr.reshape(-1, 5)
        corners: list[Any] = []
        for row in rows[:expected] if expected else rows:
            cx, cy, width, height, angle = (float(v) for v in row)
            corners.append(cxcywhr_to_corners(cx, cy, width, height, angle))
        return corners


@dataclass(frozen=True)
class DetectWriteResult:
    """Outcome of batch inference written next to ingest artifacts (Epic 3 input)."""

    detections: list[Detection]
    detections_path: Path
    overlay_paths: list[Path] = field(default_factory=list)
    n_frames: int = 0
    video_id: str | None = None


def list_frame_images(frames_dir: str | Path) -> list[Path]:
    """Return sorted image paths under ``frames_dir`` (non-images skipped)."""
    directory = Path(frames_dir)
    if not directory.is_dir():
        raise DetectorError(
            f"Frames directory not found: {directory}\n"
            "Ingest first: python scripts/ingest_video.py --video <clip>"
        )
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def frame_index_from_path(path: str | Path, *, fallback: int = 0) -> int:
    """Read a trailing integer from ``frame_000012.jpg`` (or ``A_frame_0001``)."""
    match = _FRAME_INDEX_RE.search(Path(path).stem)
    return int(match.group(1)) if match else int(fallback)


def write_detections_json(
    detections: Sequence[Detection],
    dest: str | Path,
    *,
    video_id: str | None = None,
    weights: str | None = None,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    imgsz: int | None = None,
    n_frames: int | None = None,
) -> Path:
    """Write FR-DET detection objects plus run metadata for Epic 3.

    The file is ``data/processed/<video_id>/detections.json``. Each row in
    ``detections`` matches the FR-DET example (``class``, corners, cxcywhr).
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "weights": weights,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "imgsz": imgsz,
        "n_frames": n_frames if n_frames is not None else 0,
        "n_detections": len(detections),
        "detections": [det.to_dict() for det in detections],
    }
    dest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest_path


def detect_and_write(
    detector: VehicleDetector,
    frame_paths: Sequence[str | Path],
    dest: str | Path,
    *,
    video_id: str | None = None,
    overlay_dir: str | Path | None = None,
    weights: str | None = None,
) -> DetectWriteResult:
    """Run :meth:`VehicleDetector.detect_objects` on ingested frames and persist JSON.

    Optional ``overlay_dir`` writes qualitative pred polygons
    (``data/processed/<video_id>/det_overlays/``).
    """
    paths = [Path(p) for p in frame_paths]
    detections: list[Detection] = []
    overlay_paths: list[Path] = []
    overlay_root = Path(overlay_dir) if overlay_dir is not None else None

    write_overlay = None
    if overlay_root is not None:
        from computer_vision.detection.evaluator import write_obb_overlay

        write_overlay = write_obb_overlay
        overlay_root.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(paths):
        frame_index = frame_index_from_path(image_path, fallback=i)
        dets = detector.detect_objects(image_path, frame_index=frame_index)
        detections.extend(dets)
        if write_overlay is None or overlay_root is None:
            continue
        overlay_paths.extend(
            _write_frame_overlay(write_overlay, image_path, overlay_root, dets)
        )

    dest_path = write_detections_json(
        detections,
        dest,
        video_id=video_id,
        weights=weights if weights is not None else str(detector.weights),
        conf_threshold=detector.conf_threshold,
        iou_threshold=detector.iou_threshold,
        imgsz=detector.imgsz,
        n_frames=len(paths),
    )
    return DetectWriteResult(
        detections=detections,
        detections_path=dest_path,
        overlay_paths=overlay_paths,
        n_frames=len(paths),
        video_id=video_id,
    )


def detect_video(
    detector: VehicleDetector,
    video_path: str | Path,
    dest: str | Path,
    *,
    video_id: str | None = None,
    overlay_dir: str | Path | None = None,
    weights: str | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> DetectWriteResult:
    """Decode a video, run OBB inference, and write ``detections.json``.

    Frame indices are the source-video indices (same convention as ingest).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    import cv2

    path = Path(video_path)
    if not path.is_file():
        raise DetectorError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DetectorError(f"Unable to open video: {path}")

    write_overlay = None
    overlay_root = Path(overlay_dir) if overlay_dir is not None else None
    if overlay_root is not None:
        from computer_vision.detection.evaluator import write_obb_overlay

        write_overlay = write_obb_overlay
        overlay_root.mkdir(parents=True, exist_ok=True)

    detections: list[Detection] = []
    overlay_paths: list[Path] = []
    n_kept = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                dets = detector.detect_objects(frame, frame_index=frame_idx)
                detections.extend(dets)
                if write_overlay is not None and overlay_root is not None:
                    dest_img = overlay_root / f"frame_{frame_idx:06d}.jpg"
                    overlay_paths.append(write_overlay(frame, dest_img, dets))
                n_kept += 1
                if max_frames is not None and n_kept >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    dest_path = write_detections_json(
        detections,
        dest,
        video_id=video_id,
        weights=weights if weights is not None else str(detector.weights),
        conf_threshold=detector.conf_threshold,
        iou_threshold=detector.iou_threshold,
        imgsz=detector.imgsz,
        n_frames=n_kept,
    )
    return DetectWriteResult(
        detections=detections,
        detections_path=dest_path,
        overlay_paths=overlay_paths,
        n_frames=n_kept,
        video_id=video_id,
    )


def _write_frame_overlay(
    write_overlay: Any,
    image_path: Path,
    overlay_root: Path,
    detections: Sequence[Detection],
) -> list[Path]:
    """Load one ingested frame and write a pred-only overlay; skip unreadable files."""
    import cv2

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        return []
    dest = overlay_root / f"{image_path.stem}.jpg"
    return [write_overlay(frame, dest, detections)]


def _tensor_to_numpy(value: Any) -> Any:
    """Detach a torch tensor if needed; leave numpy / lists unchanged."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return value
