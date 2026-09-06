"""Tests for detect_frames helpers and CLI (no GPU / no real weights)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from computer_vision.detection.detector import (
    DetectorError,
    VehicleDetector,
    detect_and_write,
    detect_video,
    frame_index_from_path,
    list_frame_images,
    write_detections_json,
)
from computer_vision.detection.types import Detection

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeOBB:
    def __init__(self, *, xyxyxyxy: Any, conf: Any, cls: Any) -> None:
        self.xyxyxyxy = xyxyxyxy
        self.conf = conf
        self.cls = cls


class _FakeResult:
    def __init__(self, obb: Any) -> None:
        self.obb = obb


class _StubYOLO:
    def __init__(self, obb: _FakeOBB | None) -> None:
        self.obb = obb
        self.predict_calls = 0

    def predict(self, source: Any, **kwargs: Any) -> list[_FakeResult]:
        self.predict_calls += 1
        return [_FakeResult(self.obb)]


def _sample_det() -> Detection:
    return Detection.from_cxcywhr(
        frame=12,
        class_id=1,
        confidence=0.94,
        center_x=160.0,
        center_y=230.0,
        width=80.0,
        height=60.0,
        angle=0.0,
    )


def _stub_one_car() -> _StubYOLO:
    corners = np.array(
        [[[120.0, 200.0], [200.0, 200.0], [200.0, 260.0], [120.0, 260.0]]],
        dtype=np.float64,
    )
    return _StubYOLO(
        _FakeOBB(xyxyxyxy=corners, conf=np.array([0.94]), cls=np.array([1]))
    )


def _write_tiny_jpeg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return path


def _load_detect_frames_module() -> Any:
    """Load scripts/detect_frames.py as a module (scripts/ is not a package)."""
    path = _REPO_ROOT / "scripts" / "detect_frames.py"
    spec = importlib.util.spec_from_file_location("detect_frames_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_index_from_path() -> None:
    """Ingest names and DRIFT-style stems expose a trailing frame index."""
    assert frame_index_from_path(Path("frame_000012.jpg")) == 12
    assert frame_index_from_path(Path("A_frame_0001.png")) == 1
    assert frame_index_from_path(Path("no_digits.jpg"), fallback=7) == 7


def test_list_frame_images_skips_non_images(tmp_path: Path) -> None:
    """Only image suffixes are returned; missing dirs raise DetectorError."""
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "notes.txt").write_text("skip", encoding="utf-8")
    _write_tiny_jpeg(frames / "frame_000001.jpg")
    _write_tiny_jpeg(frames / "frame_000000.jpg")
    listed = list_frame_images(frames)
    assert [p.name for p in listed] == ["frame_000000.jpg", "frame_000001.jpg"]
    with pytest.raises(DetectorError, match="Frames directory not found"):
        list_frame_images(tmp_path / "absent")


def test_write_detections_json_matches_fr_det_fields(tmp_path: Path) -> None:
    """detections.json wraps FR-DET objects plus Epic 3 run metadata."""
    dest = tmp_path / "processed" / "clip" / "detections.json"
    written = write_detections_json(
        [_sample_det()],
        dest,
        video_id="clip",
        weights="models/your_obb.pt",
        conf_threshold=0.25,
        iou_threshold=0.7,
        imgsz=640,
        n_frames=1,
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["video_id"] == "clip"
    assert payload["n_detections"] == 1
    assert payload["n_frames"] == 1
    row = payload["detections"][0]
    assert row["frame"] == 12
    assert row["class_id"] == 1
    assert row["class"] == "car"
    assert row["confidence"] == pytest.approx(0.94)
    assert len(row["corners"]) == 4


def test_detect_and_write_stub_and_overlays(tmp_path: Path) -> None:
    """Stub detector writes detections.json and optional det_overlays/."""
    frames = tmp_path / "site_03_clip_01" / "frames"
    image = _write_tiny_jpeg(frames / "frame_000003.jpg")
    dest = tmp_path / "site_03_clip_01" / "detections.json"
    overlay_dir = tmp_path / "site_03_clip_01" / "det_overlays"
    detector = VehicleDetector(model=_stub_one_car())

    result = detect_and_write(
        detector,
        [image],
        dest,
        video_id="site_03_clip_01",
        overlay_dir=overlay_dir,
        weights="models/your_obb.pt",
    )

    assert result.n_frames == 1
    assert result.video_id == "site_03_clip_01"
    assert dest.is_file()
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["detections"][0]["frame"] == 3
    assert payload["detections"][0]["class"] == "car"
    assert len(result.overlay_paths) == 1
    assert result.overlay_paths[0].is_file()
    loaded = cv2.imread(str(result.overlay_paths[0]), cv2.IMREAD_COLOR)
    assert loaded is not None


def test_detect_video_stub(tmp_path: Path) -> None:
    """detect_video decodes a synthetic clip and writes detections.json."""
    video = tmp_path / "site_03_clip_01.mp4"
    width, height, frames, fps = 64, 48, 6, 10.0
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((height, width, 3), i, dtype=np.uint8))
    writer.release()

    dest = tmp_path / "site_03_clip_01" / "detections.json"
    stub = _stub_one_car()
    result = detect_video(
        VehicleDetector(model=stub),
        video,
        dest,
        video_id="site_03_clip_01",
        stride=2,
        max_frames=2,
    )
    assert result.n_frames == 2
    assert stub.predict_calls == 2
    assert dest.is_file()
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["n_frames"] == 2
    frames_seen = {row["frame"] for row in payload["detections"]}
    assert frames_seen == {0, 2}


def test_cli_dry_run_no_source(capsys: pytest.CaptureFixture[str]) -> None:
    """detect_frames.py --dry-run with no source still exits 0."""
    main = _load_detect_frames_module().main
    code = main(["--dry-run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Dry run OK" in captured.out
    assert "not running inference" in captured.out


def test_cli_dry_run_with_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run --frames reports image count and does not write detections.json."""
    frames = tmp_path / "clip" / "frames"
    _write_tiny_jpeg(frames / "frame_000000.jpg")
    dest = tmp_path / "clip" / "detections.json"
    main = _load_detect_frames_module().main
    code = main(
        [
            "--frames",
            str(frames),
            "--out",
            str(dest),
            "--weights",
            str(tmp_path / "your_obb.pt"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Dry run OK" in captured.out
    assert "1 images" in captured.out
    assert not dest.exists()


def test_cli_missing_weights_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --dry-run, missing your_obb.pt is exit 1 (NFR-ACC-004)."""
    frames = tmp_path / "clip" / "frames"
    _write_tiny_jpeg(frames / "frame_000000.jpg")
    main = _load_detect_frames_module().main
    code = main(
        [
            "--frames",
            str(frames),
            "--out",
            str(tmp_path / "detections.json"),
            "--weights",
            str(tmp_path / "your_obb.pt"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "OBB weights not found" in captured.err
    assert "FR-DET-001" in captured.err


def test_cli_missing_frames_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing frames directory fails clearly before inference."""
    main = _load_detect_frames_module().main
    code = main(
        [
            "--frames",
            str(tmp_path / "no_frames"),
            "--weights",
            str(tmp_path / "your_obb.pt"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "Frames directory not found" in captured.err


def test_cli_writes_detections_with_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI persist path writes detections.json when VehicleDetector is stubbed."""
    frames = tmp_path / "clip" / "frames"
    _write_tiny_jpeg(frames / "frame_000005.jpg")
    dest = tmp_path / "clip" / "detections.json"
    module = _load_detect_frames_module()
    stub = _stub_one_car()

    def fake_detector(*args: Any, **kwargs: Any) -> VehicleDetector:
        return VehicleDetector(
            model=stub,
            conf_threshold=kwargs.get("conf_threshold", 0.25),
            iou_threshold=kwargs.get("iou_threshold", 0.7),
            imgsz=kwargs.get("imgsz", 640),
            device=kwargs.get("device", "cpu"),
        )

    monkeypatch.setattr(module, "VehicleDetector", fake_detector)
    code = module.main(
        ["--frames", str(frames), "--out", str(dest), "--video-id", "clip"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert dest.is_file()
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["video_id"] == "clip"
    assert payload["detections"][0]["frame"] == 5
    assert payload["detections"][0]["class"] == "car"
    assert "detections.json" in captured.out


def test_cli_no_source_without_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real run without --video-id / --frames / --video is exit 1."""
    main = _load_detect_frames_module().main
    code = main([])
    captured = capsys.readouterr()
    assert code == 1
    assert "Specify --video-id, --frames, or --video" in captured.err
