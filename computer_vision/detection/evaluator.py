"""Evaluate Traffilytics YOLO OBB weights against held-out labels (FR-DET-006).

``DetectionEvaluator.evaluate`` runs Ultralytics ``model.val`` when weights (or
an injected stub) are available and writes ``metrics.json`` under
``models/runs/eval_<name>/``. Qualitative overlays are always drawable from
``list[Detection]`` vs GT :class:`~adapters.drift.obb_annotations.OBBBox`
polygons (OpenCV ``polylines``) — no GPU or checkpoint required for that path.

Do not call real Ultralytics ``train()`` from pytest; inject ``model`` instead.
Missing product weights fail with a clear error (NFR-ACC-004) unless this is a
dry-run (path checks only) or ``allow_pretrained`` is set for CPU wiring.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from adapters.drift.obb_annotations import OBBBox, load_obb_label_file
from computer_vision.detection.detector import VehicleDetector
from computer_vision.detection.types import (
    CLASS_NAMES,
    Detection,
    as_corners,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = _REPO_ROOT / "models" / "your_obb.pt"
DEFAULT_DATA_YAML = _REPO_ROOT / "models" / "configs" / "drift_obb_data.yaml"
DEFAULT_PROJECT = _REPO_ROOT / "models" / "runs"
DEFAULT_EVAL_NAME = "obb"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_IMGSZ = 640
PRETRAINED_WEIGHTS = "yolo11n-obb.pt"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_PRETRAINED_WARNING = (
    "DetectionEvaluator is using pretrained yolo11n-obb.pt, not models/your_obb.pt. "
    "This is CPU/wiring only and does not satisfy FR-DET-001."
)

# BGR colors for qualitative overlays (prediction tint by class; GT is green).
_GT_COLOR = (60, 220, 60)
_PRED_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 140, 0),
    1: (0, 220, 255),
    2: (200, 80, 255),
}
_DEFAULT_PRED_COLOR = (0, 165, 255)
_TEXT_COLOR = (255, 255, 255)
_TEXT_SCALE = 0.45
_LINE_THICKNESS = 2


class EvaluatorError(Exception):
    """Raised when eval weights, dataset, or overlay I/O fail (NFR-ACC-004)."""

    exit_code = 1


def _load_yolo(weights: str) -> Any:
    """Import Ultralytics and construct ``YOLO(weights)``.

    Isolated so tests can monkeypatch without importing torch.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise EvaluatorError(
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


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    """Turn a repo-relative path into an absolute path under ``repo_root``."""
    path = Path(value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def _jsonish(value: Any) -> Any:
    """Coerce Ultralytics metric values into JSON-friendly Python types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Mapping):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    return str(value)


def _raw_metrics(val_result: Any) -> dict[str, Any]:
    """Pull a flat dict out of Ultralytics ``val`` output (or a stub mapping)."""
    if val_result is None:
        return {}
    if isinstance(val_result, Mapping):
        return {str(k): _jsonish(v) for k, v in val_result.items()}
    results_dict = getattr(val_result, "results_dict", None)
    if isinstance(results_dict, Mapping):
        return {str(k): _jsonish(v) for k, v in results_dict.items()}
    box = getattr(val_result, "box", None)
    if box is not None:
        out: dict[str, Any] = {}
        for key in ("map50", "map", "map75"):
            if hasattr(box, key):
                out[key] = _jsonish(getattr(box, key))
        maps = getattr(box, "maps", None)
        if maps is not None:
            out["maps"] = _jsonish(maps)
        if out:
            return out
    return {"repr": repr(val_result)}


def _alias_map_keys(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Expose ``mAP50`` / ``mAP50-95`` beside Ultralytics' raw key names."""
    aliases: dict[str, Any] = {}
    pairs = (
        ("mAP50", ("metrics/mAP50(B)", "metrics/mAP50(P)", "map50")),
        ("mAP50-95", ("metrics/mAP50-95(B)", "metrics/mAP50-95(P)", "map")),
    )
    for dest, sources in pairs:
        for src in sources:
            if src in raw and raw[src] is not None:
                aliases[dest] = raw[src]
                break
    return aliases


def _per_class_maps(raw: Mapping[str, Any], val_result: Any) -> dict[str, Any] | None:
    """Map per-class AP values onto bus/car/truck names when Ultralytics provides them."""
    maps = raw.get("maps")
    if maps is None:
        box = getattr(val_result, "box", None)
        maps = getattr(box, "maps", None) if box is not None else None
        maps = _jsonish(maps) if maps is not None else None
    if not isinstance(maps, (list, tuple)):
        return None
    return {CLASS_NAMES.get(i, str(i)): _jsonish(v) for i, v in enumerate(maps)}


def normalize_val_metrics(val_result: Any) -> dict[str, Any]:
    """Normalize Ultralytics ``val`` output into a JSON-serializable metrics dict.

    Always includes ``raw``. Adds ``mAP50``, ``mAP50-95``, and ``per_class`` when
    those values can be recovered from the result object or a stub mapping.
    """
    raw = _raw_metrics(val_result)
    out: dict[str, Any] = {"raw": raw}
    out.update(_alias_map_keys(raw))
    per_class = _per_class_maps(raw, val_result)
    if per_class:
        out["per_class"] = per_class
    return out


def _box_class_id(box: Any) -> int:
    """Read ``class_id`` from an :class:`OBBBox`, :class:`Detection`, or similar."""
    return int(box.class_id)


def _box_class_name(box: Any) -> str:
    """Human-readable class label for a GT or prediction box."""
    name = getattr(box, "class_name", None)
    if name:
        return str(name)
    return CLASS_NAMES.get(_box_class_id(box), str(_box_class_id(box)))


def _box_confidence(box: Any) -> float | None:
    """Return detection confidence when present (GT boxes have none)."""
    if isinstance(box, Detection):
        return float(box.confidence)
    conf = getattr(box, "confidence", None)
    if conf is None:
        return None
    return float(conf)


def corners_to_pixel_pts(
    corners: Sequence[Any],
    frame_width: int,
    frame_height: int,
    *,
    normalized: bool | None = None,
) -> np.ndarray:
    """Convert four corners to an ``(4, 1, 2)`` int32 contour for ``cv2.polylines``.

    Adapter GT labels are normalized; inference :class:`Detection` corners are
    pixels. When ``normalized`` is omitted, values entirely in ``[0, 1]`` are
    treated as normalized.
    """
    pts = np.asarray(as_corners(corners), dtype=np.float64)
    if normalized is None:
        normalized = bool(pts.size and np.all((pts >= 0.0) & (pts <= 1.0)))
    if normalized:
        pts = pts.copy()
        pts[:, 0] *= float(frame_width)
        pts[:, 1] *= float(frame_height)
    return np.round(pts).astype(np.int32).reshape((-1, 1, 2))


def _as_bgr_canvas(frame: np.ndarray) -> np.ndarray:
    """Return a writable BGR ``uint8`` copy of ``frame`` (does not mutate input)."""
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy ndarray, got {type(frame).__name__}")
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        canvas = frame.copy()
    else:
        raise ValueError(f"Frame must be H×W or H×W×3, got shape {frame.shape}")
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


def _draw_one_box(
    canvas: np.ndarray,
    corners: Sequence[Any],
    *,
    color: tuple[int, int, int],
    label: str,
    normalized: bool | None,
) -> None:
    """Draw one closed OBB polygon and a class label on ``canvas``."""
    height, width = canvas.shape[:2]
    pts = corners_to_pixel_pts(corners, width, height, normalized=normalized)
    cv2.polylines(
        canvas,
        [pts],
        isClosed=True,
        color=color,
        thickness=_LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    if not label:
        return
    flat = pts.reshape(-1, 2)
    top = flat[int(np.argmin(flat[:, 1]))]
    origin = (int(top[0]), max(12, int(top[1]) - 4))
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


def draw_obb_overlay(
    frame: np.ndarray,
    detections: Sequence[Detection] = (),
    ground_truth: Sequence[OBBBox | Detection] | None = None,
) -> np.ndarray:
    """Draw prediction (class-colored) and GT (green) OBB polygons on a frame copy.

    Works with no model and no weights. GT :class:`OBBBox` corners are treated
    as normalized; :class:`Detection` corners default to pixels unless every
    value lies in ``[0, 1]``.
    """
    canvas = _as_bgr_canvas(frame)
    for box in ground_truth or ():
        _draw_one_box(
            canvas,
            box.corners,
            color=_GT_COLOR,
            label=f"{_box_class_name(box)} GT",
            normalized=None if isinstance(box, Detection) else True,
        )
    for det in detections:
        color = _PRED_COLORS.get(det.class_id, _DEFAULT_PRED_COLOR)
        conf = _box_confidence(det)
        label = det.class_name if conf is None else f"{det.class_name} {conf:.2f}"
        _draw_one_box(
            canvas,
            det.corners,
            color=color,
            label=label,
            normalized=None,
        )
    return canvas


def write_obb_overlay(
    frame: np.ndarray,
    dest: str | Path,
    detections: Sequence[Detection] = (),
    ground_truth: Sequence[OBBBox | Detection] | None = None,
) -> Path:
    """Draw an overlay and write it to ``dest`` (parent dirs created as needed)."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = draw_obb_overlay(frame, detections, ground_truth)
    suffix = dest_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(
            str(dest_path),
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
    else:
        ok = cv2.imwrite(str(dest_path), canvas)
    if not ok:
        raise EvaluatorError(f"Failed to write overlay image: {dest_path}")
    return dest_path


def write_metrics_json(payload: Mapping[str, Any], dest: str | Path) -> Path:
    """Write a metrics (or comparison) report as indented JSON."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(
        json.dumps(_jsonish(dict(payload)), indent=2) + "\n",
        encoding="utf-8",
    )
    return dest_path


def _eval_run_name(name: str) -> str:
    """Prefix ``eval_`` unless the caller already supplied that folder name."""
    cleaned = str(name).strip() or DEFAULT_EVAL_NAME
    return cleaned if cleaned.startswith("eval_") else f"eval_{cleaned}"


def _load_data_yaml(path: Path) -> dict[str, Any]:
    """Load an Ultralytics data.yaml mapping or raise :class:`EvaluatorError`."""
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, Mapping):
        raise EvaluatorError(f"data.yaml must be a mapping: {path}")
    return dict(loaded)


def _dataset_root(data_yaml: Path, cfg: Mapping[str, Any]) -> Path:
    """Resolve the dataset root listed in data.yaml (``path`` or YAML parent)."""
    raw = cfg.get("path")
    if raw is None:
        return data_yaml.parent.resolve()
    root = Path(str(raw))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def _split_images_dir(root: Path, cfg: Mapping[str, Any], split: str) -> Path | None:
    """Return the images directory for ``split`` (``val`` falls back to ``valid``)."""
    candidates = [split]
    if split == "val":
        candidates.extend(("valid", "validation"))
    elif split == "valid":
        candidates.append("val")
    rel: str | None = None
    for key in candidates:
        if cfg.get(key):
            rel = str(cfg[key])
            break
    if rel is None:
        return None
    images_dir = Path(rel)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    return images_dir


def iter_heldout_pairs(
    data_yaml: Path,
    split: str = "val",
) -> Iterable[tuple[Path, Path | None]]:
    """Yield ``(image_path, label_path|None)`` for the held-out YOLO split.

    Expects the usual ``images/`` + sibling ``labels/`` layout. Missing image
    directories yield nothing (overlays are skipped, val metrics still run).
    """
    cfg = _load_data_yaml(data_yaml)
    root = _dataset_root(data_yaml, cfg)
    images_dir = _split_images_dir(root, cfg, split)
    if images_dir is None or not images_dir.is_dir():
        return
    labels_dir = images_dir.parent / "labels"
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        yield image_path, label_path if label_path.is_file() else None


def _load_gt_boxes(label_path: Path | None) -> list[OBBBox]:
    """Read YOLO-OBB labels; missing or empty files become an empty list."""
    if label_path is None or not label_path.is_file():
        return []
    return load_obb_label_file(label_path)


@dataclass(frozen=True)
class EvalPlan:
    """Resolved eval paths and settings after existence checks (no Ultralytics)."""

    weights: Path
    data_yaml: Path
    run_dir: Path
    metrics_path: Path
    overlay_dir: Path
    device: str
    split: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    max_overlays: int
    weights_missing: bool
    weights_arg: str
    pretrained_warning: str | None
    baseline: Path | None


@dataclass(frozen=True)
class EvalResult:
    """Outcome of :meth:`DetectionEvaluator.evaluate`, including dry-run.

    ``metrics`` is empty on dry-run. After a real eval it holds normalized
    product-model mAP fields. ``baseline_metrics`` is set only when a baseline
    checkpoint was compared (never used as the product model).
    """

    plan: EvalPlan
    dry_run: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    metrics_path: Path | None = None
    overlay_dir: Path | None = None
    overlay_paths: list[Path] = field(default_factory=list)
    baseline_metrics: dict[str, Any] | None = None
    report: dict[str, Any] = field(default_factory=dict)


class DetectionEvaluator:
    """Held-out mAP (Ultralytics ``val``) plus qualitative OBB overlays.

    Inject ``model`` to stub Ultralytics in tests. Overlay drawing
    (:meth:`draw_overlay`, :func:`draw_obb_overlay`) never requires weights.
    Optional ``baseline`` compares DRIFT ``best.pt`` in the metrics JSON only —
    product inference still uses Traffilytics weights.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        data: str | Path | None = None,
        name: str = DEFAULT_EVAL_NAME,
        project: str | Path | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = "auto",
        split: str = "val",
        max_overlays: int = 32,
        allow_pretrained: bool = False,
        baseline: str | Path | None = None,
        repo_root: str | Path | None = None,
        model: Any | None = None,
        baseline_model: Any | None = None,
    ) -> None:
        """Configure an eval run; Ultralytics is not imported until :meth:`evaluate`.

        ``model`` / ``baseline_model`` skip weight-file loads (tests). Relative
        paths resolve against ``repo_root`` (the Traffilytics checkout).
        """
        if not 0.0 <= float(conf_threshold) <= 1.0:
            raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold}")
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if int(imgsz) <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}")
        if int(max_overlays) < 0:
            raise ValueError(f"max_overlays must be >= 0, got {max_overlays}")

        self.repo_root = Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
        self.weights = (
            Path(weights) if weights is not None else DEFAULT_WEIGHTS
        )
        self._data = data if data is not None else DEFAULT_DATA_YAML
        self.name = str(name)
        self.project = Path(project) if project is not None else DEFAULT_PROJECT
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.split = str(split)
        self.max_overlays = int(max_overlays)
        self.allow_pretrained = bool(allow_pretrained)
        self._baseline = Path(baseline) if baseline is not None else None
        self._model = model
        self._baseline_model = baseline_model

    def resolve(self) -> EvalPlan:
        """Resolve paths and device; require the dataset YAML (no Ultralytics)."""
        data_yaml = _resolve_path(self._data, self.repo_root)
        if not data_yaml.is_file():
            raise EvaluatorError(
                f"data.yaml not found: {data_yaml}\n"
                "Run: python scripts/prepare_obb_dataset.py"
            )

        weights = self.weights
        if not weights.is_absolute():
            weights = _resolve_path(weights, self.repo_root)
        else:
            weights = weights.resolve()

        weights_missing = not weights.is_file()
        pretrained_warning: str | None = None
        if self._model is not None:
            weights_arg = str(weights)
        elif not weights_missing:
            weights_arg = str(weights)
        elif self.allow_pretrained:
            pretrained_warning = _PRETRAINED_WARNING
            weights_arg = PRETRAINED_WEIGHTS
        else:
            weights_arg = str(weights)

        project = self.project
        if not project.is_absolute():
            project = _resolve_path(project, self.repo_root)
        run_dir = project / _eval_run_name(self.name)

        baseline = self._baseline
        if baseline is not None:
            if not baseline.is_absolute():
                baseline = _resolve_path(baseline, self.repo_root)
            else:
                baseline = baseline.resolve()

        return EvalPlan(
            weights=weights,
            data_yaml=data_yaml,
            run_dir=run_dir,
            metrics_path=run_dir / "metrics.json",
            overlay_dir=run_dir / "overlays",
            device=_resolve_device(self.device),
            split=self.split,
            imgsz=self.imgsz,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_overlays=self.max_overlays,
            weights_missing=weights_missing and self._model is None,
            weights_arg=weights_arg,
            pretrained_warning=pretrained_warning,
            baseline=baseline,
        )

    def evaluate(
        self,
        *,
        dry_run: bool = False,
        write_overlays: bool = True,
    ) -> EvalResult:
        """Run held-out ``val``, write ``metrics.json``, and optional overlays.

        Dry-run validates the dataset YAML (and notes missing weights) without
        calling Ultralytics. A real run requires product weights, an injected
        ``model``, or ``allow_pretrained``.

        Raises:
            EvaluatorError: missing data.yaml, missing weights, or overlay I/O.
        """
        plan = self.resolve()
        if plan.pretrained_warning:
            warnings.warn(plan.pretrained_warning, UserWarning, stacklevel=2)

        if dry_run:
            return EvalResult(plan=plan, dry_run=True)

        if plan.weights_missing and not self.allow_pretrained:
            raise EvaluatorError(
                f"Traffilytics OBB weights not found: {plan.weights}. "
                "Train on a CUDA host with: python scripts/train_obb.py "
                "(copies best.pt to models/your_obb.pt). "
                "For CPU wiring only, pass allow_pretrained=True to use "
                "yolo11n-obb.pt — that is not FR-DET-001 compliant."
            )

        model = self._model if self._model is not None else _load_yolo(plan.weights_arg)
        val_kwargs = self._val_kwargs(plan)
        metrics = normalize_val_metrics(model.val(**val_kwargs))

        baseline_metrics: dict[str, Any] | None = None
        if plan.baseline is not None or self._baseline_model is not None:
            baseline_metrics = self._evaluate_baseline(plan, val_kwargs)

        overlay_paths: list[Path] = []
        overlay_dir: Path | None = None
        if write_overlays:
            overlay_dir = plan.overlay_dir
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_paths = self._write_split_overlays(plan, model)

        report = {
            "weights": plan.weights_arg,
            "data": str(plan.data_yaml),
            "split": plan.split,
            "device": plan.device,
            "imgsz": plan.imgsz,
            "conf_threshold": plan.conf_threshold,
            "iou_threshold": plan.iou_threshold,
            "metrics": metrics,
            "baseline": (
                None
                if baseline_metrics is None
                else {
                    "weights": str(plan.baseline) if plan.baseline else None,
                    "metrics": baseline_metrics,
                }
            ),
            "overlay_dir": str(overlay_dir) if overlay_dir is not None else None,
            "n_overlays": len(overlay_paths),
        }
        metrics_path = write_metrics_json(report, plan.metrics_path)
        return EvalResult(
            plan=plan,
            dry_run=False,
            metrics=metrics,
            metrics_path=metrics_path,
            overlay_dir=overlay_dir,
            overlay_paths=overlay_paths,
            baseline_metrics=baseline_metrics,
            report=report,
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection] = (),
        ground_truth: Sequence[OBBBox | Detection] | None = None,
    ) -> np.ndarray:
        """Draw prediction and GT polygons on a copy of ``frame`` (no weights)."""
        return draw_obb_overlay(frame, detections, ground_truth)

    def write_overlay(
        self,
        frame: np.ndarray,
        dest: str | Path,
        detections: Sequence[Detection] = (),
        ground_truth: Sequence[OBBBox | Detection] | None = None,
    ) -> Path:
        """Draw and persist one overlay image (no weights)."""
        return write_obb_overlay(frame, dest, detections, ground_truth)

    def _val_kwargs(self, plan: EvalPlan) -> dict[str, Any]:
        """Keyword arguments shared by product and baseline ``model.val`` calls."""
        return {
            "data": str(plan.data_yaml),
            "split": plan.split,
            "imgsz": plan.imgsz,
            "conf": plan.conf_threshold,
            "iou": plan.iou_threshold,
            "device": plan.device,
            "verbose": False,
            "plots": False,
        }

    def _evaluate_baseline(
        self,
        plan: EvalPlan,
        val_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run ``val`` on the optional DRIFT ``best.pt`` comparison checkpoint."""
        if self._baseline_model is not None:
            baseline_model = self._baseline_model
        else:
            if plan.baseline is None or not plan.baseline.is_file():
                raise EvaluatorError(
                    f"Baseline weights not found: {plan.baseline}. "
                    "Pass a real DRIFT best.pt for comparison only."
                )
            baseline_model = _load_yolo(str(plan.baseline))
        return normalize_val_metrics(baseline_model.val(**dict(val_kwargs)))

    def _write_split_overlays(self, plan: EvalPlan, model: Any) -> list[Path]:
        """Detect on held-out frames and write qualitative pred-vs-GT overlays."""
        if plan.max_overlays == 0:
            return []

        detector = VehicleDetector(
            plan.weights,
            conf_threshold=plan.conf_threshold,
            iou_threshold=plan.iou_threshold,
            imgsz=plan.imgsz,
            device=self.device,
            allow_pretrained=self.allow_pretrained,
            model=model,
        )
        written: list[Path] = []
        for frame_index, (image_path, label_path) in enumerate(
            iter_heldout_pairs(plan.data_yaml, plan.split)
        ):
            if len(written) >= plan.max_overlays:
                break
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            detections = detector.detect_objects(frame, frame_index=frame_index)
            dest = plan.overlay_dir / f"{image_path.stem}.jpg"
            written.append(
                write_obb_overlay(
                    frame,
                    dest,
                    detections,
                    _load_gt_boxes(label_path),
                )
            )
        return written
