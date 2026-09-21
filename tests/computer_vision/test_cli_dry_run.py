"""Subprocess CPU tests for train/eval/detect CLIs (no GPU, no Ultralytics train()).

These cover the Step 2 "Done when" commands:

- ``python scripts/train_obb.py --dry-run``
- ``python scripts/eval_obb.py --dry-run``
- ``python scripts/detect_frames.py --dry-run``
- ``python scripts/track_video.py --dry-run``

plus missing-weights error paths. Timeouts fail fast if a real ``train()`` starts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"
_DATA_YAML = _REPO_ROOT / "models" / "configs" / "uav_obb_data.yaml"
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
                "  0: bike",
                "  1: bus",
                "  2: car",
                "  3: other_vehicle",
                "  4: taxi",
                "  5: truck",
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


@pytest.mark.skipif(not _DATA_YAML.is_file(), reason="generated uav_obb_data.yaml not present")
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


@pytest.mark.skipif(not _DATA_YAML.is_file(), reason="generated uav_obb_data.yaml not present")
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


def test_track_video_script_dry_run(tmp_path: Path) -> None:
    """``python scripts/track_video.py --dry-run`` validates detections.json on CPU."""
    from computer_vision.detection.detector import write_detections_json
    from computer_vision.detection.types import Detection

    dest_dir = tmp_path / "clip"
    detections_path = dest_dir / "detections.json"
    det = Detection.from_cxcywhr(
        frame=0,
        class_id=2,
        confidence=0.9,
        center_x=50.0,
        center_y=50.0,
        width=20.0,
        height=10.0,
        angle=0.0,
    )
    write_detections_json([det], detections_path, video_id="clip", n_frames=1)
    out = dest_dir / "trajectories.json"
    result = _run_script(
        "track_video.py",
        [
            "--config",
            str(_DEFAULT_CONFIG),
            "--detections",
            str(detections_path),
            "--out",
            str(out),
            "--dry-run",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run OK" in result.stdout
    assert "not starting tracking" in result.stdout
    assert not out.exists()


def test_track_video_script_missing_detections(tmp_path: Path) -> None:
    result = _run_script(
        "track_video.py",
        [
            "--config",
            str(_DEFAULT_CONFIG),
            "--detections",
            str(tmp_path / "missing.json"),
        ],
    )
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()


def test_track_video_script_writes_trajectories(tmp_path: Path) -> None:
    """Real ByteTrack path writes trajectories.json (skipped without ultralytics)."""
    pytest.importorskip("ultralytics")
    from computer_vision.detection.detector import write_detections_json
    from computer_vision.detection.types import Detection

    dets = [
        Detection.from_cxcywhr(
            frame=i,
            class_id=2,
            confidence=0.9,
            center_x=80.0 + i * 12.0,
            center_y=100.0,
            width=40.0,
            height=20.0,
            angle=0.0,
        )
        for i in range(8)
    ]
    detections_path = tmp_path / "clip" / "detections.json"
    write_detections_json(dets, detections_path, video_id="clip", n_frames=8)
    out = tmp_path / "clip" / "trajectories.json"
    result = _run_script(
        "track_video.py",
        [
            "--config",
            str(_DEFAULT_CONFIG),
            "--detections",
            str(detections_path),
            "--out",
            str(out),
        ],
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tracker"] == "bytetrack"
    assert payload["n_tracks"] == 1
    assert payload["points"][0]["lane"] is None
