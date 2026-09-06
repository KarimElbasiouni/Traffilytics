"""CPU tests for download_obb_dataset.py — local copy only, no GitHub fetch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_main():
    """Load scripts/download_obb_dataset.py:main without installing the script as a package."""
    path = _REPO_ROOT / "scripts" / "download_obb_dataset.py"
    spec = importlib.util.spec_from_file_location("download_obb_dataset_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _write_github_model_tree(root: Path) -> Path:
    """Create a tiny The-DRIFT-style model/{train,valid,test} tree."""
    model = root / "model"
    for split, stem in (("train", "A_frame_0000"), ("valid", "A_frame_0001")):
        images = model / split / "images"
        labels = model / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{stem}.jpg").write_bytes(b"fake")
        (labels / f"{stem}.txt").write_text(
            "1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n", encoding="utf-8"
        )
    (model / "data.yaml").write_text("names: {1: car}\n", encoding="utf-8")
    return model


def test_cli_dry_run_without_full_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run documents the ~2,301-frame set and does not require --src."""
    main = _load_main()
    code = main(["--dry-run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "2,301" in captured.out
    assert "300,000" in captured.out or "300K" in captured.out
    assert "Dry run OK" in captured.out
    assert "do not download" in captured.out.lower()


def test_cli_refuses_src_without_full(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--src without --full exits 1 so a huge copy cannot happen by accident."""
    src = tmp_path / "The-DRIFT"
    _write_github_model_tree(src)
    main = _load_main()
    code = main(["--src", str(src), "--dest", str(tmp_path / "annotations")])
    captured = capsys.readouterr()
    assert code == 1
    assert "without --full" in captured.err
    assert not (tmp_path / "annotations" / "train").exists()


def test_cli_full_requires_src(capsys: pytest.CaptureFixture[str]) -> None:
    """--full with no --src exits 1 and does not clone GitHub."""
    main = _load_main()
    code = main(["--full"])
    captured = capsys.readouterr()
    assert code == 1
    assert "--src" in captured.err
    assert "does not download" in captured.err


def test_cli_full_dry_run_does_not_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--full --dry-run --src prints the plan and leaves dest empty."""
    src = tmp_path / "The-DRIFT"
    dest = tmp_path / "annotations"
    _write_github_model_tree(src)
    main = _load_main()
    code = main(["--full", "--dry-run", "--src", str(src), "--dest", str(dest)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Dry run OK" in captured.out
    assert str(src / "model") in captured.out
    assert not dest.exists() or not any(dest.iterdir())


def test_cli_full_copies_local_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--full --src copies train/valid via sync_github_style_layout (no network)."""
    src = tmp_path / "The-DRIFT"
    dest = tmp_path / "annotations"
    _write_github_model_tree(src)
    main = _load_main()
    code = main(["--full", "--src", str(src), "--dest", str(dest)])
    captured = capsys.readouterr()
    assert code == 0
    assert (dest / "train" / "images" / "A_frame_0000.jpg").is_file()
    assert (dest / "valid" / "labels" / "A_frame_0001.txt").is_file()
    assert (dest / "data.yaml").is_file()
    assert "prepare_obb_dataset.py" in captured.out
    assert "WARNING" in captured.err  # tiny fixture is not the 2,301-frame set


def test_cli_full_missing_src_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--full --src on an empty directory exits 1 with a clear error."""
    main = _load_main()
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main(["--full", "--src", str(empty), "--dest", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 1
    assert "ERROR" in captured.err
    assert "model/" in captured.err
