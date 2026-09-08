"""Subprocess CPU tests for train/eval/detect CLIs (no GPU, no Ultralytics train()).

These cover the Step 2 "Done when" commands:

- ``python scripts/train_obb.py --dry-run``
- ``python scripts/eval_obb.py --dry-run``
- ``python scripts/detect_frames.py --dry-run``

plus missing-weights error paths. Timeouts fail fast if a real ``train()`` starts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"
_DATA_YAML = _REPO_ROOT / "models" / "configs" / "drift_obb_data.yaml"
_PRODUCT_WEIGHTS = _REPO_ROOT / "models" / "your_obb.pt"
_CLI_TIMEOUT_S = 60


def _run_script(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``scripts/<script>`` with the pytest interpreter from the repo root."""
    return subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / script), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_S,
        check=False,
    )


def _write_data_yaml(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                f"path: {path.parent}",
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


def _write_train_yaml(path: Path, *, data_yaml: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "model: yolo11n-obb.pt",
                f"data: {data_yaml}",
                "epochs: 1",
                "imgsz: 640",
                "batch: 1",
                "device: 0",
                "workers: 0",
                "project: models/runs",
                "name: obb_cli_test",
                "exist_ok: true",
                "patience: 1",
                "save: false",
                "plots: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_tiny_jpeg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return path


def test_train_obb_script_dry_run(tmp_path: Path) -> None:
    """``python scripts/train_obb.py --dry-run`` exits 0 and does not train."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    train_yaml = _write_train_yaml(tmp_path / "train.yaml", data_yaml=data_yaml)
    result = _run_script(
        "train_obb.py",
        ["--train-config", str(train_yaml), "--dry-run"],
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert "not starting training" in result.stdout
    assert not (tmp_path / "your_obb.pt").exists()
    assert not (_REPO_ROOT / "models" / "runs" / "obb_cli_test" / "weights" / "best.pt").exists()


@pytest.mark.skipif(not _DATA_YAML.is_file(), reason="generated drift_obb_data.yaml not present")
def test_train_obb_default_dry_run() -> None:
    """Operator command ``python scripts/train_obb.py --dry-run`` works on CPU."""
    existed = _PRODUCT_WEIGHTS.is_file()
    mtime = _PRODUCT_WEIGHTS.stat().st_mtime if existed else None
    result = _run_script("train_obb.py", ["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert "not starting training" in result.stdout
    if existed:
        assert _PRODUCT_WEIGHTS.stat().st_mtime == mtime
    else:
        assert not _PRODUCT_WEIGHTS.exists()


def test_eval_obb_script_dry_run(tmp_path: Path) -> None:
    """``python scripts/eval_obb.py --dry-run`` exits 0 and does not write metrics."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    result = _run_script(
        "eval_obb.py",
        [
            "--config",
            str(_DEFAULT_CONFIG),
            "--data",
            str(data_yaml),
            "--weights",
            str(tmp_path / "missing.pt"),
            "--project",
            str(tmp_path / "runs"),
            "--dry-run",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert "not starting evaluation" in result.stdout
    assert not (tmp_path / "runs" / "eval_obb" / "metrics.json").exists()


@pytest.mark.skipif(not _DATA_YAML.is_file(), reason="generated drift_obb_data.yaml not present")
def test_eval_obb_default_dry_run() -> None:
    """Operator command ``python scripts/eval_obb.py --dry-run`` works on CPU."""
    result = _run_script(
        "eval_obb.py",
        ["--config", str(_DEFAULT_CONFIG), "--dry-run"],
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert "not starting evaluation" in result.stdout


def test_eval_obb_script_missing_weights(tmp_path: Path) -> None:
    """Without --dry-run, missing your_obb.pt is exit 1 (NFR-ACC-004)."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    result = _run_script(
        "eval_obb.py",
        [
            "--config",
            str(_DEFAULT_CONFIG),
            "--data",
            str(data_yaml),
            "--weights",
            str(tmp_path / "your_obb.pt"),
            "--project",
            str(tmp_path / "runs"),
        ],
    )
    assert result.returncode == 1
    assert "OBB weights not found" in result.stderr
    assert "FR-DET-001" in result.stderr
    assert not (tmp_path / "runs" / "eval_obb" / "metrics.json").exists()


def test_detect_frames_script_dry_run() -> None:
    """Operator command ``python scripts/detect_frames.py --dry-run`` works on CPU."""
    result = _run_script(
        "detect_frames.py",
        ["--config", str(_DEFAULT_CONFIG), "--dry-run"],
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert "not running inference" in result.stdout


def test_detect_frames_script_dry_run_with_frames(tmp_path: Path) -> None:
    """--dry-run --frames reports the image count and does not write detections.json."""
    frames = tmp_path / "clip" / "frames"
    _write_tiny_jpeg(frames / "frame_000000.jpg")
    dest = tmp_path / "clip" / "detections.json"
    result = _run_script(
        "detect_frames.py",
        [
            "--config",
            str(_DEFAULT_CONFIG),
            "--frames",
            str(frames),
            "--out",
            str(dest),
            "--weights",
            str(tmp_path / "your_obb.pt"),
            "--dry-run",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert "1 images" in result.stdout
    assert not dest.exists()


def test_detect_frames_script_missing_weights(tmp_path: Path) -> None:
    """Without --dry-run, missing your_obb.pt is exit 1 (NFR-ACC-004)."""
    frames = tmp_path / "clip" / "frames"
    _write_tiny_jpeg(frames / "frame_000000.jpg")
    dest = tmp_path / "clip" / "detections.json"
    result = _run_script(
        "detect_frames.py",
        [
            "--config",
            str(_DEFAULT_CONFIG),
            "--frames",
            str(frames),
            "--out",
            str(dest),
            "--weights",
            str(tmp_path / "your_obb.pt"),
        ],
    )
    assert result.returncode == 1
    assert "OBB weights not found" in result.stderr
    assert "FR-DET-001" in result.stderr
    assert not dest.exists()
