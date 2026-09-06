"""Tests for DRIFT OBB annotation adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adapters.drift.obb_annotations import (
    DriftOBBDataset,
    install_github_obb_splits,
    load_obb_label_file,
    parse_obb_label_line,
    resolve_github_model_dir,
)


def test_parse_obb_label_line() -> None:
    """A valid 9-field OBB line becomes an OBBBox with class car and 8 corners."""
    line = "1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4"
    box = parse_obb_label_line(line)
    assert box.class_id == 1
    assert box.class_name == "car"
    assert len(box.corners) == 8


def test_parse_obb_label_line_rejects_bad() -> None:
    """Malformed label lines (wrong field count) raise ValueError."""
    with pytest.raises(ValueError):
        parse_obb_label_line("1 0.1 0.2")


def test_dataset_adapter_and_yaml(tmp_path: Path) -> None:
    """DriftOBBDataset finds train/val pairs and writes a usable Ultralytics data.yaml."""
    train_img = tmp_path / "train" / "images"
    train_lbl = tmp_path / "train" / "labels"
    val_img = tmp_path / "valid" / "images"
    val_lbl = tmp_path / "valid" / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True)

    (train_img / "A_frame_0000.jpg").write_bytes(b"fake")
    (train_lbl / "A_frame_0000.txt").write_text(
        "1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n", encoding="utf-8"
    )
    (val_img / "A_frame_0001.jpg").write_bytes(b"fake")
    (val_lbl / "A_frame_0001.txt").write_text(
        "0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n", encoding="utf-8"
    )

    ds = DriftOBBDataset.from_annotations_root(tmp_path)
    assert ds.count_pairs("train") == 1
    assert ds.count_pairs("val") == 1
    boxes = load_obb_label_file(train_lbl / "A_frame_0000.txt")
    assert boxes[0].class_id == 1

    out = tmp_path / "data.yaml"
    ds.write_ultralytics_data_yaml(out)
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert payload["train"] == "train/images"
    assert payload["names"][1] == "car"


def _write_github_model_tree(root: Path, *, nest_under_model: bool = False) -> Path:
    """Create a tiny GitHub-style train/valid/test tree; return the model/ dir."""
    model = root / "model" if nest_under_model else root
    for split, stem, class_id in (
        ("train", "A_frame_0000", "1"),
        ("valid", "A_frame_0001", "0"),
        ("test", "A_frame_0002", "2"),
    ):
        images = model / split / "images"
        labels = model / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{stem}.jpg").write_bytes(b"fake")
        (labels / f"{stem}.txt").write_text(
            f"{class_id} 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n", encoding="utf-8"
        )
    (model / "data.yaml").write_text("names: {0: bus}\n", encoding="utf-8")
    return model


def test_resolve_github_model_dir_accepts_repo_root_or_model(tmp_path: Path) -> None:
    """Repo root (…/The-DRIFT) and its model/ folder both resolve to the split tree."""
    model = _write_github_model_tree(tmp_path, nest_under_model=True)
    assert resolve_github_model_dir(tmp_path) == model
    assert resolve_github_model_dir(model) == model


def test_resolve_github_model_dir_rejects_empty(tmp_path: Path) -> None:
    """A directory without train/valid raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="model/"):
        resolve_github_model_dir(tmp_path)


def test_sync_and_install_github_obb_splits(tmp_path: Path) -> None:
    """install_github_obb_splits copies train/valid/test + data.yaml into dest."""
    repo = tmp_path / "The-DRIFT"
    dest = tmp_path / "annotations"
    _write_github_model_tree(repo, nest_under_model=True)

    written = install_github_obb_splits(repo, dest)
    assert written == dest.resolve()
    assert (dest / "train" / "images" / "A_frame_0000.jpg").is_file()
    assert (dest / "valid" / "labels" / "A_frame_0001.txt").is_file()
    assert (dest / "test" / "images" / "A_frame_0002.jpg").is_file()
    assert (dest / "data.yaml").is_file()

    ds = DriftOBBDataset.from_annotations_root(dest)
    assert ds.count_pairs("train") == 1
    assert ds.count_pairs("val") == 1
    assert ds.count_pairs("test") == 1

    # Same dest is a no-op (already in place).
    again = install_github_obb_splits(dest, dest)
    assert again == dest.resolve()
