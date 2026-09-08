"""Tests for DetectionEvaluator: overlays, metrics JSON, and stubbed val (no GPU)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = str(_REPO_ROOT / "configs" / "default.yaml")

from adapters.drift.obb_annotations import OBBBox
from computer_vision.detection.evaluator import (
    DetectionEvaluator,
    EvaluatorError,
    draw_obb_overlay,
    normalize_val_metrics,
    write_obb_overlay,
)
from computer_vision.detection.types import Detection, cxcywhr_to_corners


class _FakeOBB:
    """Minimal Ultralytics-style OBB container for overlay-inference stubs."""

    def __init__(self, *, xyxyxyxy: Any, conf: Any, cls: Any) -> None:
        self.xyxyxyxy = xyxyxyxy
        self.conf = conf
        self.cls = cls


class _FakeResult:
    def __init__(self, obb: Any) -> None:
        self.obb = obb


class _StubYOLO:
    """Ultralytics stand-in: records val kwargs and returns canned mAP + OBBs."""

    def __init__(
        self,
        *,
        map50: float = 0.55,
        map5095: float = 0.30,
        maps: list[float] | None = None,
        corners: np.ndarray | None = None,
    ) -> None:
        self.map50 = map50
        self.map5095 = map5095
        self.maps = maps if maps is not None else [0.4, 0.6, 0.5]
        self.corners = corners
        self.val_calls = 0
        self.predict_calls = 0
        self.last_val_kwargs: dict[str, Any] | None = None

    def val(self, **kwargs: Any) -> dict[str, Any]:
        self.val_calls += 1
        self.last_val_kwargs = kwargs
        return {
            "metrics/mAP50(B)": self.map50,
            "metrics/mAP50-95(B)": self.map5095,
            "maps": list(self.maps),
        }

    def predict(self, source: Any, **kwargs: Any) -> list[_FakeResult]:
        self.predict_calls += 1
        if self.corners is None:
            return [_FakeResult(None)]
        n = len(self.corners)
        obb = _FakeOBB(
            xyxyxyxy=self.corners,
            conf=np.full((n,), 0.9, dtype=np.float64),
            cls=np.arange(n) % 3,
        )
        return [_FakeResult(obb)]


def _write_data_yaml(path: Path, *, dataset_root: Path | None = None) -> Path:
    root = dataset_root if dataset_root is not None else path.parent
    path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: train/images",
                "val: valid/images",
                "names:",
                "  0: bus",
                "  1: car",
                "  2: truck",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _pixel_car() -> Detection:
    """Axis-aligned car box in pixel coordinates (matches FR-DET example geometry)."""
    return Detection.from_cxcywhr(
        frame=0,
        class_id=1,
        confidence=0.94,
        center_x=160.0,
        center_y=230.0,
        width=80.0,
        height=60.0,
        angle=0.0,
    )


def _normalized_gt_car() -> OBBBox:
    """GT car covering the same region as ``_pixel_car`` on a 320×320 frame."""
    # Pixel box (120,200)–(200,260) → normalized on 320×320.
    return OBBBox(
        class_id=1,
        corners=(
            120 / 320,
            200 / 320,
            200 / 320,
            200 / 320,
            200 / 320,
            260 / 320,
            120 / 320,
            260 / 320,
        ),
    )


def test_draw_overlay_writes_image_for_synthetic_frame(tmp_path: Path) -> None:
    """Evaluator overlay writes an image for a synthetic frame + pred/GT boxes."""
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detections = [_pixel_car()]
    gt = [_normalized_gt_car()]

    canvas = draw_obb_overlay(frame, detections, gt)
    assert canvas.shape == frame.shape
    assert canvas is not frame
    assert not np.array_equal(canvas, frame)

    dest = tmp_path / "overlays" / "frame_0000.jpg"
    evaluator = DetectionEvaluator(model=_StubYOLO(), data=tmp_path / "unused.yaml")
    written = evaluator.write_overlay(frame, dest, detections, gt)

    assert written == dest
    assert dest.is_file()
    loaded = cv2.imread(str(dest), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert loaded.shape == (320, 320, 3)
    assert not np.array_equal(loaded, frame)


def test_draw_overlay_does_not_require_weights(tmp_path: Path) -> None:
    """Polygon overlays work with no checkpoint and no Ultralytics model."""
    frame = np.full((64, 64, 3), 10, dtype=np.uint8)
    det = Detection(0, 0, 0.8, cxcywhr_to_corners(32.0, 32.0, 20.0, 12.0, 0.3))
    path = write_obb_overlay(frame, tmp_path / "only_pred.png", [det], None)
    assert path.is_file()
    loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert not np.array_equal(loaded, frame)


def test_gt_normalized_corners_are_scaled() -> None:
    """Normalized GT corners are scaled to frame pixels before drawing."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    gt = [
        OBBBox(
            class_id=2,
            corners=(0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8),
        )
    ]
    canvas = draw_obb_overlay(frame, [], gt)
    # The scaled square occupies the center; edges around (20,20)–(80,80) are painted.
    assert canvas[20, 50].sum() > 0
    assert canvas[50, 20].sum() > 0
    assert canvas[0, 0].sum() == 0


def test_normalize_val_metrics_aliases_and_per_class() -> None:
    """Ultralytics-style keys become mAP50 / mAP50-95 plus per-class names."""
    raw = {
        "metrics/mAP50(B)": 0.71,
        "metrics/mAP50-95(B)": 0.44,
        "maps": [0.1, 0.2, 0.3],
    }
    out = normalize_val_metrics(raw)
    assert out["mAP50"] == pytest.approx(0.71)
    assert out["mAP50-95"] == pytest.approx(0.44)
    assert out["per_class"] == {"bus": 0.1, "car": 0.2, "truck": 0.3}
    assert out["raw"]["metrics/mAP50(B)"] == pytest.approx(0.71)


def test_evaluate_writes_metrics_json(tmp_path: Path) -> None:
    """Stub val() writes models/runs/eval_<name>/metrics.json with mAP fields."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    stub = _StubYOLO()
    evaluator = DetectionEvaluator(
        weights=tmp_path / "your_obb.pt",
        data=data_yaml,
        name="obb_test",
        project=tmp_path / "runs",
        repo_root=tmp_path,
        model=stub,
        max_overlays=0,
    )
    result = evaluator.evaluate(write_overlays=False)

    assert result.dry_run is False
    assert stub.val_calls == 1
    assert stub.last_val_kwargs is not None
    assert stub.last_val_kwargs["data"] == str(data_yaml)
    assert stub.last_val_kwargs["split"] == "val"
    assert result.metrics["mAP50"] == pytest.approx(0.55)
    assert result.metrics["mAP50-95"] == pytest.approx(0.30)
    assert result.metrics_path is not None and result.metrics_path.is_file()
    assert result.metrics_path == tmp_path / "runs" / "eval_obb_test" / "metrics.json"

    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["mAP50"] == pytest.approx(0.55)
    assert payload["metrics"]["per_class"]["car"] == pytest.approx(0.6)
    assert payload["n_overlays"] == 0
    assert payload["baseline"] is None


def test_evaluate_writes_overlays_from_heldout_split(tmp_path: Path) -> None:
    """evaluate() draws pred vs GT overlays for readable val-split images."""
    images = tmp_path / "valid" / "images"
    labels = tmp_path / "valid" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.imwrite(str(images / "A_frame_0001.jpg"), frame)
    labels.joinpath("A_frame_0001.txt").write_text(
        "1 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    data_yaml = _write_data_yaml(tmp_path / "data.yaml", dataset_root=tmp_path)

    corners = np.array(
        [[[10.0, 10.0], [30.0, 10.0], [30.0, 24.0], [10.0, 24.0]]],
        dtype=np.float64,
    )
    stub = _StubYOLO(corners=corners)
    evaluator = DetectionEvaluator(
        weights=tmp_path / "your_obb.pt",
        data=data_yaml,
        name="split",
        project=tmp_path / "runs",
        repo_root=tmp_path,
        model=stub,
        max_overlays=4,
    )
    result = evaluator.evaluate()

    assert stub.val_calls == 1
    assert stub.predict_calls == 1
    assert result.overlay_dir == tmp_path / "runs" / "eval_split" / "overlays"
    assert len(result.overlay_paths) == 1
    overlay = result.overlay_paths[0]
    assert overlay.is_file()
    assert overlay.name == "A_frame_0001.jpg"
    loaded = cv2.imread(str(overlay), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert not np.array_equal(loaded, frame)


def test_evaluate_dry_run_skips_val_and_files(tmp_path: Path) -> None:
    """Dry-run checks the data YAML and does not call val() or write reports."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    stub = _StubYOLO()
    evaluator = DetectionEvaluator(
        weights=tmp_path / "missing.pt",
        data=data_yaml,
        project=tmp_path / "runs",
        repo_root=tmp_path,
        model=stub,
    )
    result = evaluator.evaluate(dry_run=True)

    assert result.dry_run is True
    assert result.metrics == {}
    assert result.metrics_path is None
    assert stub.val_calls == 0
    assert not (tmp_path / "runs" / "eval_obb" / "metrics.json").exists()
    assert result.plan.data_yaml == data_yaml


def test_evaluate_missing_weights_error(tmp_path: Path) -> None:
    """Real evaluate() without your_obb.pt fails clearly (NFR-ACC-004)."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    evaluator = DetectionEvaluator(
        weights=tmp_path / "your_obb.pt",
        data=data_yaml,
        repo_root=tmp_path,
    )
    with pytest.raises(EvaluatorError, match="OBB weights not found"):
        evaluator.evaluate()
    with pytest.raises(EvaluatorError, match="FR-DET-001"):
        evaluator.evaluate()


def test_evaluate_missing_data_yaml(tmp_path: Path) -> None:
    """Missing dataset YAML is a clear EvaluatorError before Ultralytics."""
    evaluator = DetectionEvaluator(
        weights=tmp_path / "your_obb.pt",
        data=tmp_path / "absent.yaml",
        repo_root=tmp_path,
        model=_StubYOLO(),
    )
    with pytest.raises(EvaluatorError, match="data.yaml not found"):
        evaluator.evaluate(dry_run=True)


def test_evaluate_baseline_side_by_side(tmp_path: Path) -> None:
    """Baseline val() is recorded next to product metrics; not used as product."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    product = _StubYOLO(map50=0.80, map5095=0.50)
    baseline = _StubYOLO(map50=0.60, map5095=0.35)
    evaluator = DetectionEvaluator(
        weights=tmp_path / "your_obb.pt",
        data=data_yaml,
        name="cmp",
        project=tmp_path / "runs",
        repo_root=tmp_path,
        model=product,
        baseline=tmp_path / "best.pt",
        baseline_model=baseline,
        max_overlays=0,
    )
    result = evaluator.evaluate(write_overlays=False)

    assert product.val_calls == 1
    assert baseline.val_calls == 1
    assert result.metrics["mAP50"] == pytest.approx(0.80)
    assert result.baseline_metrics is not None
    assert result.baseline_metrics["mAP50"] == pytest.approx(0.60)
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert payload["metrics"]["mAP50"] == pytest.approx(0.80)
    assert payload["baseline"]["metrics"]["mAP50"] == pytest.approx(0.60)


def test_evaluate_allow_pretrained_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """allow_pretrained loads yolo11n-obb.pt with a loud FR-DET-001 warning."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    seen: dict[str, str] = {}
    stub = _StubYOLO()

    def fake_load(weights: str) -> _StubYOLO:
        seen["weights"] = weights
        return stub

    monkeypatch.setattr(
        "computer_vision.detection.evaluator._load_yolo", fake_load
    )
    evaluator = DetectionEvaluator(
        weights=tmp_path / "missing.pt",
        data=data_yaml,
        project=tmp_path / "runs",
        repo_root=tmp_path,
        allow_pretrained=True,
        max_overlays=0,
    )
    with pytest.warns(UserWarning, match="FR-DET-001"):
        result = evaluator.evaluate(write_overlays=False)
    assert seen["weights"] == "yolo11n-obb.pt"
    assert result.metrics["mAP50"] == pytest.approx(0.55)


def test_evaluator_rejects_bad_conf_threshold(tmp_path: Path) -> None:
    """Constructor rejects conf_threshold outside [0, 1]."""
    with pytest.raises(ValueError, match="conf_threshold"):
        DetectionEvaluator(conf_threshold=1.5, data=tmp_path / "data.yaml")


def _load_eval_obb_main() -> Any:
    """Load scripts/eval_obb.py as a module (scripts/ is not a package)."""
    path = _REPO_ROOT / "scripts" / "eval_obb.py"
    spec = importlib.util.spec_from_file_location("eval_obb_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_cli_dry_run_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """eval_obb.py --dry-run prints Dry run OK and does not write metrics."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    main = _load_eval_obb_main()
    code = main(
        [
            "--config",
            _DEFAULT_CONFIG,
            "--data",
            str(data_yaml),
            "--weights",
            str(tmp_path / "missing.pt"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Dry run OK" in captured.out
    assert "not starting evaluation" in captured.out
    assert not (tmp_path / "runs" / "eval_obb" / "metrics.json").exists()


def test_cli_dry_run_reports_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--baseline is resolved on dry-run and labeled missing when the file is absent."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    baseline = tmp_path / "best.pt"
    main = _load_eval_obb_main()
    code = main(
        [
            "--config",
            _DEFAULT_CONFIG,
            "--data",
            str(data_yaml),
            "--weights",
            str(tmp_path / "your_obb.pt"),
            "--baseline",
            str(baseline),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Baseline" in captured.out
    assert "missing" in captured.out
    assert "best.pt" in captured.out


def test_cli_missing_weights_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --dry-run, missing your_obb.pt is exit 1 (NFR-ACC-004)."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    main = _load_eval_obb_main()
    code = main(
        [
            "--config",
            _DEFAULT_CONFIG,
            "--data",
            str(data_yaml),
            "--weights",
            str(tmp_path / "your_obb.pt"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "OBB weights not found" in captured.err
    assert "FR-DET-001" in captured.err


def test_cli_missing_data_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing dataset YAML returns exit 1 with a clear error."""
    main = _load_eval_obb_main()
    code = main(
        [
            "--config",
            _DEFAULT_CONFIG,
            "--data",
            str(tmp_path / "absent.yaml"),
            "--weights",
            str(tmp_path / "your_obb.pt"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "data.yaml not found" in captured.err
