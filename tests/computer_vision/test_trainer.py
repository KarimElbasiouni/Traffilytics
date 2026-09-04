"""Tests for DetectionTrainer: CUDA gate, dry-run, and stubbed Ultralytics (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from computer_vision.detection.trainer import (
    CudaUnavailableError,
    DetectionTrainer,
    TrainerError,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _StubTrainResults:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = str(save_dir)


class _StubYOLO:
    """Ultralytics stand-in: records train/val kwargs and never touches CUDA."""

    def __init__(self, save_dir: Path, *, write_best: bool = True) -> None:
        self.save_dir = save_dir
        self.write_best = write_best
        self.train_calls = 0
        self.val_calls = 0
        self.last_train_kwargs: dict[str, Any] | None = None
        self.last_val_kwargs: dict[str, Any] | None = None

    def train(self, **kwargs: Any) -> _StubTrainResults:
        self.train_calls += 1
        self.last_train_kwargs = kwargs
        if self.write_best:
            best = self.save_dir / "weights" / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"fake-best-weights")
        return _StubTrainResults(self.save_dir)

    def val(self, **kwargs: Any) -> dict[str, float]:
        self.val_calls += 1
        self.last_val_kwargs = kwargs
        return {"metrics/mAP50(B)": 0.42, "metrics/mAP50-95(B)": 0.21}


def _write_data_yaml(path: Path) -> Path:
    path.write_text("path: .\ntrain: train/images\nval: valid/images\n", encoding="utf-8")
    return path


def _write_train_yaml(path: Path, *, data_yaml: Path, device: str = "0") -> Path:
    path.write_text(
        "\n".join(
            [
                "model: yolo11n-obb.pt",
                f"data: {data_yaml}",
                "epochs: 3",
                "imgsz: 640",
                "batch: 2",
                f"device: {device}",
                "workers: 0",
                "project: models/runs",
                "name: obb_test",
                "exist_ok: true",
                "patience: 5",
                "save: true",
                "plots: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _trainer(
    tmp_path: Path,
    *,
    device: str | None = None,
    model: Any | None = None,
    cfg_device: str = "0",
) -> DetectionTrainer:
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    train_yaml = _write_train_yaml(
        tmp_path / "train.yaml", data_yaml=data_yaml, device=cfg_device
    )
    return DetectionTrainer(
        train_yaml,
        data=None,
        device=device,
        weights_dest=tmp_path / "your_obb.pt",
        repo_root=tmp_path,
        model=model,
    )


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend torch is installed but this host has no GPU."""
    monkeypatch.setattr(
        "computer_vision.detection.trainer._cuda_status",
        lambda: (False, "2.0.0+cpu"),
    )


@pytest.fixture
def has_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "computer_vision.detection.trainer._cuda_status",
        lambda: (True, "2.0.0+cu118"),
    )


def test_dry_run_does_not_call_ultralytics(tmp_path: Path, no_cuda: None) -> None:
    """Dry-run validates paths and returns a metrics stub without model.train()."""
    stub = _StubYOLO(tmp_path / "runs" / "obb_test")
    trainer = _trainer(tmp_path, model=stub)
    result = trainer.train(dry_run=True)

    assert result.dry_run is True
    assert result.metrics == {}
    assert result.product_weights is None
    assert result.run_dir == tmp_path / "models" / "runs" / "obb_test"
    assert stub.train_calls == 0
    assert stub.val_calls == 0
    assert result.plan.cuda_warning is not None
    assert result.plan.model_name == "yolo11n-obb.pt"


def test_cuda_gate_refuses_config_gpu_without_cuda(
    tmp_path: Path, no_cuda: None
) -> None:
    """Config device 0 and no CUDA → refuse real training (exit 2)."""
    stub = _StubYOLO(tmp_path / "runs" / "obb_test")
    trainer = _trainer(tmp_path, model=stub)
    with pytest.raises(CudaUnavailableError, match="Refusing to start GPU training") as exc:
        trainer.train(dry_run=False)
    assert exc.value.exit_code == 2
    assert exc.value.warning is not None
    assert "CUDA is not available" in exc.value.warning
    assert stub.train_calls == 0


def test_cuda_gate_allows_explicit_cpu_smoke(tmp_path: Path, no_cuda: None) -> None:
    """--device cpu on a CPU box is a smoke run, not a refuse."""
    save_dir = tmp_path / "models" / "runs" / "obb_test"
    stub = _StubYOLO(save_dir)
    trainer = _trainer(tmp_path, device="cpu", model=stub)
    result = trainer.train(dry_run=False)

    assert result.dry_run is False
    assert stub.train_calls == 1
    assert stub.last_train_kwargs is not None
    assert stub.last_train_kwargs["device"] == "cpu"
    assert result.product_weights == tmp_path / "your_obb.pt"
    assert result.product_weights is not None and result.product_weights.is_file()
    assert result.product_weights.read_bytes() == b"fake-best-weights"
    assert result.metrics["metrics/mAP50(B)"] == pytest.approx(0.42)


def test_explicit_gpu_without_cuda_falls_back_to_cpu(
    tmp_path: Path, no_cuda: None
) -> None:
    """CLI --device 0 without CUDA warns and trains on CPU instead of refusing."""
    save_dir = tmp_path / "models" / "runs" / "obb_test"
    stub = _StubYOLO(save_dir)
    trainer = _trainer(tmp_path, device="0", model=stub)
    result = trainer.train(dry_run=False)

    assert result.plan.cuda_warning is not None
    assert stub.last_train_kwargs is not None
    assert stub.last_train_kwargs["device"] == "cpu"
    assert result.plan.train_device == "cpu"


def test_missing_data_yaml(tmp_path: Path, no_cuda: None) -> None:
    """Missing dataset YAML is a clear TrainerError (NFR-ACC-004)."""
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text("model: yolo11n-obb.pt\ndata: missing.yaml\n", encoding="utf-8")
    trainer = DetectionTrainer(train_yaml, repo_root=tmp_path, model=_StubYOLO(tmp_path))
    with pytest.raises(TrainerError, match="data.yaml not found"):
        trainer.train(dry_run=True)


def test_missing_train_config(tmp_path: Path, no_cuda: None) -> None:
    """Missing train YAML fails before Ultralytics is imported."""
    trainer = DetectionTrainer(tmp_path / "nope.yaml", repo_root=tmp_path)
    with pytest.raises(TrainerError, match="Missing train config"):
        trainer.train(dry_run=True)


def test_train_copies_best_pt_and_returns_run_dir(
    tmp_path: Path, has_cuda: None
) -> None:
    """After stub train(), best.pt is copied to models/your_obb.pt."""
    save_dir = tmp_path / "models" / "runs" / "obb_test"
    stub = _StubYOLO(save_dir)
    trainer = _trainer(tmp_path, model=stub)
    result = trainer.train()

    assert result.run_dir == save_dir
    assert result.best_weights == save_dir / "weights" / "best.pt"
    dest = tmp_path / "your_obb.pt"
    assert dest.is_file()
    assert stub.last_train_kwargs is not None
    assert stub.last_train_kwargs["data"] == str(tmp_path / "data.yaml")
    assert stub.last_val_kwargs is not None
    assert stub.val_calls == 1


def test_missing_best_pt_does_not_fail(tmp_path: Path, has_cuda: None) -> None:
    """If Ultralytics did not write best.pt, training still returns (with a stub)."""
    save_dir = tmp_path / "models" / "runs" / "obb_test"
    stub = _StubYOLO(save_dir, write_best=False)
    trainer = _trainer(tmp_path, model=stub)
    result = trainer.train()
    assert result.product_weights is None
    assert result.best_weights is None
    assert not (tmp_path / "your_obb.pt").exists()
    assert result.metrics


def _load_train_obb_main() -> Any:
    """Load scripts/train_obb.py as a module (scripts/ is not a package)."""
    path = _REPO_ROOT / "scripts" / "train_obb.py"
    spec = importlib.util.spec_from_file_location("train_obb_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_cli_dry_run_ok(tmp_path: Path, no_cuda: None, capsys: pytest.CaptureFixture[str]) -> None:
    """train_obb.py --dry-run prints Dry run OK and does not train."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    train_yaml = _write_train_yaml(tmp_path / "train.yaml", data_yaml=data_yaml)
    main = _load_train_obb_main()
    code = main(["--train-config", str(train_yaml), "--dry-run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Dry run OK" in captured.out
    assert "WARNING" in captured.out
    assert "not starting training" in captured.out


def test_cli_cuda_refuse_exit_2(
    tmp_path: Path, no_cuda: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --dry-run or --device cpu, the CLI refuses GPU training (exit 2)."""
    data_yaml = _write_data_yaml(tmp_path / "data.yaml")
    train_yaml = _write_train_yaml(tmp_path / "train.yaml", data_yaml=data_yaml)
    main = _load_train_obb_main()
    code = main(["--train-config", str(train_yaml)])
    captured = capsys.readouterr()
    assert code == 2
    assert "WARNING" in captured.out
    assert "Refusing to start GPU training" in captured.err


def test_cli_missing_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing train YAML returns exit 1 with a clear error."""
    main = _load_train_obb_main()
    code = main(["--train-config", str(tmp_path / "absent.yaml"), "--dry-run"])
    captured = capsys.readouterr()
    assert code == 1
    assert "Missing train config" in captured.err
