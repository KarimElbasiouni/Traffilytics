"""Train Traffilytics YOLO OBB weights from a dataset YAML (FR-DET-005).

``DetectionTrainer`` wraps Ultralytics ``YOLO.train`` and copies ``best.pt`` to
``models/your_obb.pt`` for product inference. Practical training is a CUDA-host
operator step: without a GPU, ``train()`` refuses unless ``device`` is explicitly
``cpu`` (tiny smoke) or ``dry_run`` is True (path/config checks only).

Do not call Ultralytics ``train()`` from pytest — inject ``model`` instead.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = _REPO_ROOT / "models" / "configs" / "train_obb.yaml"
DEFAULT_DATA_YAML = _REPO_ROOT / "models" / "configs" / "drift_obb_data.yaml"
DEFAULT_WEIGHTS_DEST = _REPO_ROOT / "models" / "your_obb.pt"
DEFAULT_MODEL = "yolov8n-obb.pt"
_CUDA_REFUSE_MESSAGE = (
    "Refusing to start GPU training without CUDA. "
    "Re-run with --device cpu (smoke only) or on a GPU machine."
)


class TrainerError(Exception):
    """Raised when training config, dataset, or runtime checks fail (NFR-ACC-004)."""

    exit_code = 1


class CudaUnavailableError(TrainerError):
    """GPU device requested but CUDA is not available, and this is not a dry-run."""

    exit_code = 2

    def __init__(self, message: str, *, warning: str | None = None) -> None:
        super().__init__(message)
        self.warning = warning
        self.exit_code = 2


def _cuda_status() -> tuple[bool, str]:
    """Return ``(cuda_available, torch_version)``. Isolated for tests to monkeypatch."""
    try:
        import torch
    except ImportError as exc:
        raise TrainerError("PyTorch is not installed.") from exc
    return bool(torch.cuda.is_available()), str(torch.__version__)


def _load_yolo(model_name: str) -> Any:
    """Import Ultralytics and construct ``YOLO(model_name)`` (no train() here)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainerError(
            "ultralytics is not installed. pip install ultralytics"
        ) from exc
    return YOLO(model_name)


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


def _metrics_from_val(val_result: Any) -> dict[str, Any]:
    """Normalize Ultralytics ``val`` output into a metrics stub dict."""
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


def _int_or_default(value: Any, default: int) -> int:
    """Parse a YAML number, treating missing/zero like the original script (``or``)."""
    return int(value or default)


@dataclass(frozen=True)
class TrainPlan:
    """Resolved train settings after path checks and the CUDA probe."""

    train_config_path: Path | None
    data_yaml: Path
    model_name: str
    device: str
    train_device: str
    epochs: int
    imgsz: int
    batch: int
    workers: int
    project: Path
    name: str
    exist_ok: bool
    patience: int
    save: bool
    plots: bool
    weights_dest: Path
    cuda_available: bool
    torch_version: str
    cuda_warning: str | None
    device_explicit: bool

    @property
    def run_dir(self) -> Path:
        """Ultralytics run directory (``project/name``)."""
        return self.project / self.name

    @property
    def best_pt(self) -> Path:
        """Expected ``best.pt`` path under the run directory."""
        return self.run_dir / "weights" / "best.pt"

    @property
    def device_display(self) -> str:
        """Device string for CLI logs, noting when CUDA is missing."""
        if self.cuda_available or self.device == "cpu":
            return self.device
        return f"{self.device} (CUDA missing)"


@dataclass(frozen=True)
class TrainResult:
    """Outcome of :meth:`DetectionTrainer.train`, including dry-run.

    ``metrics`` is empty on dry-run (a stub). After a real train it holds the
    quick ``model.val`` pass, when Ultralytics returns one.
    """

    plan: TrainPlan
    dry_run: bool
    run_dir: Path | None
    best_weights: Path | None
    product_weights: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)


class DetectionTrainer:
    """Train a YOLO OBB detector and publish ``models/your_obb.pt``.

    Inject ``model`` to stub Ultralytics in tests. Relative paths in the train
    YAML are resolved against ``repo_root`` (the Traffilytics checkout).
    """

    def __init__(
        self,
        train_config: str | Path | Mapping[str, Any] = DEFAULT_TRAIN_CONFIG,
        *,
        data: str | Path | None = None,
        device: str | int | None = None,
        epochs: int | None = None,
        weights_dest: str | Path | None = None,
        repo_root: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        """Configure a training run; Ultralytics is not imported until :meth:`train`.

        ``device`` ``None`` means use the YAML value (usually a GPU id). Passing
        ``device`` is treated as an explicit override: without CUDA, a non-cpu
        override warns and falls back to CPU instead of refusing. Config-default
        GPU without CUDA is refused (unless ``dry_run``).
        """
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
        self._train_config_input = train_config
        self._data_override = data
        self._device_override = device
        self._epochs_override = epochs
        self._weights_dest = (
            Path(weights_dest) if weights_dest is not None else DEFAULT_WEIGHTS_DEST
        )
        self._model = model

    def resolve(self) -> TrainPlan:
        """Load YAML, require the dataset file, and probe CUDA (no Ultralytics)."""
        cfg, config_path = self._load_config()
        data_yaml = _resolve_path(
            self._data_override or cfg.get("data") or DEFAULT_DATA_YAML,
            self.repo_root,
        )
        if not data_yaml.is_file():
            raise TrainerError(
                f"data.yaml not found: {data_yaml}\n"
                "Run: python scripts/prepare_obb_dataset.py"
            )

        device_explicit = self._device_override is not None
        raw_device = (
            self._device_override if device_explicit else cfg.get("device", "0")
        )
        device = str(raw_device)
        epochs = (
            int(self._epochs_override)
            if self._epochs_override is not None
            else _int_or_default(cfg.get("epochs"), 50)
        )

        cuda_ok, torch_version = _cuda_status()
        cuda_warning: str | None = None
        if device != "cpu" and not cuda_ok:
            cuda_warning = (
                "CUDA is not available. "
                "Set --device cpu for a tiny smoke run, or train on a CUDA host.\n"
                f"  torch={torch_version} cuda_available={cuda_ok}"
            )
        train_device = device if (cuda_ok or device == "cpu") else "cpu"

        project = _resolve_path(cfg.get("project") or "models/runs", self.repo_root)
        weights_dest = self._weights_dest
        if not weights_dest.is_absolute():
            weights_dest = _resolve_path(weights_dest, self.repo_root)

        return TrainPlan(
            train_config_path=config_path,
            data_yaml=data_yaml,
            model_name=str(cfg.get("model") or DEFAULT_MODEL),
            device=device,
            train_device=train_device,
            epochs=epochs,
            imgsz=_int_or_default(cfg.get("imgsz"), 640),
            batch=_int_or_default(cfg.get("batch"), 8),
            workers=_int_or_default(cfg.get("workers"), 4),
            project=project,
            name=str(cfg.get("name") or "obb_v1"),
            exist_ok=bool(cfg.get("exist_ok", True)),
            patience=_int_or_default(cfg.get("patience"), 20),
            save=bool(cfg.get("save", True)),
            plots=bool(cfg.get("plots", True)),
            weights_dest=weights_dest,
            cuda_available=cuda_ok,
            torch_version=torch_version,
            cuda_warning=cuda_warning,
            device_explicit=device_explicit,
        )

    def train(self, *, dry_run: bool = False) -> TrainResult:
        """Run Ultralytics training, or validate paths when ``dry_run`` is True.

        Returns a :class:`TrainResult` with the run directory and a metrics stub
        (empty on dry-run). Copies ``best.pt`` to ``models/your_obb.pt`` after a
        real run when that file exists.

        Raises:
            TrainerError: missing config/dataset, missing PyTorch/Ultralytics.
            CudaUnavailableError: GPU requested from config, no CUDA, not dry-run.
        """
        plan = self.resolve()
        if plan.cuda_warning and not dry_run and not plan.device_explicit:
            raise CudaUnavailableError(_CUDA_REFUSE_MESSAGE, warning=plan.cuda_warning)

        if dry_run:
            return TrainResult(
                plan=plan,
                dry_run=True,
                run_dir=plan.run_dir,
                best_weights=None,
                product_weights=None,
                metrics={},
            )

        model = self._model if self._model is not None else _load_yolo(plan.model_name)
        results = model.train(
            data=str(plan.data_yaml),
            epochs=plan.epochs,
            imgsz=plan.imgsz,
            batch=plan.batch,
            device=plan.train_device,
            workers=plan.workers,
            project=str(plan.project),
            name=plan.name,
            exist_ok=plan.exist_ok,
            patience=plan.patience,
            save=plan.save,
            plots=plan.plots,
        )

        best = self._find_best_weights(results, plan)
        product: Path | None = None
        if best is not None:
            plan.weights_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, plan.weights_dest)
            product = plan.weights_dest

        val_result = model.val(data=str(plan.data_yaml), device=plan.train_device)
        save_dir = getattr(results, "save_dir", None)
        run_dir = Path(save_dir) if save_dir else plan.run_dir
        return TrainResult(
            plan=plan,
            dry_run=False,
            run_dir=run_dir,
            best_weights=best,
            product_weights=product,
            metrics=_metrics_from_val(val_result),
        )

    def _load_config(self) -> tuple[dict[str, Any], Path | None]:
        """Return ``(cfg dict, yaml path or None)`` from a mapping or file."""
        raw = self._train_config_input
        if isinstance(raw, Mapping):
            return dict(raw), None

        path = _resolve_path(raw, self.repo_root)
        if not path.is_file():
            raise TrainerError(f"Missing train config: {path}")
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise TrainerError(f"Train config must be a mapping: {path}")
        return dict(loaded), path

    def _find_best_weights(self, results: Any, plan: TrainPlan) -> Path | None:
        """Locate Ultralytics ``best.pt`` from ``results.save_dir`` or the run dir."""
        save_dir = getattr(results, "save_dir", "") or ""
        candidates = [
            Path(save_dir) / "weights" / "best.pt",
            plan.best_pt,
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None
