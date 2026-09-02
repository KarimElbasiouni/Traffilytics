"""Traffilytics-trained YOLO OBB inference (FR-DET-001 … FR-DET-004).

``VehicleDetector`` wraps Ultralytics ``YOLO`` and returns :class:`Detection`
records with **pixel** corners. Product weights are ``models/your_obb.pt``
(NFR-ACC-004: missing weights fail with a clear error). ``allow_pretrained``
is a CPU/wiring escape hatch that loads ``yolo11n-obb.pt`` with a warning; it
does **not** satisfy FR-DET-001.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

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

_PRETRAINED_WARNING = (
    "VehicleDetector is using pretrained yolo11n-obb.pt, not models/your_obb.pt. "
    "This is CPU/wiring only and does not satisfy FR-DET-001."
)


class DetectorError(Exception):
    """Raised when OBB weights cannot be loaded or a frame cannot be inferred (NFR-ACC-004)."""


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


def _tensor_to_numpy(value: Any) -> Any:
    """Detach a torch tensor if needed; leave numpy / lists unchanged."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return value
