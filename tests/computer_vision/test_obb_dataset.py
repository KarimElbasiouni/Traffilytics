"""Tests for the UAV-OBB dataset adapter (no network or dataset download)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adapters.uav_obb.dataset import ObbDataset, iter_split_pairs
from computer_vision.detection.obb_labels import load_obb_label_file


def _write_split(root: Path, split: str, stem: str, class_id: str = "2") -> None:
    """Create one image/label pair under ``root/<split>/{images,labels}``."""
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / f"{stem}.jpg").write_bytes(b"fake")
    (labels / f"{stem}.txt").write_text(
        f"{class_id} 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n", encoding="utf-8"
    )


def test_dataset_adapter_and_yaml(tmp_path: Path) -> None:
    """ObbDataset finds train/valid pairs and writes a usable Ultralytics data.yaml."""
    _write_split(tmp_path, "train", "clip_0000", "2")
    _write_split(tmp_path, "valid", "clip_0001", "1")

    ds = ObbDataset.from_annotations_root(tmp_path)
    assert ds.count_pairs("train") == 1
    assert ds.count_pairs("val") == 1
    assert ds.test_dir is None

    boxes = load_obb_label_file(tmp_path / "train" / "labels" / "clip_0000.txt")
    assert boxes[0].class_name == "car"

    out = tmp_path / "data.yaml"
    ds.write_ultralytics_data_yaml(out)
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert payload["path"] == str(tmp_path.resolve())
    assert payload["train"] == "train/images"
    assert payload["val"] == "valid/images"
    assert "test" not in payload
    assert len(payload["names"]) == 6
    assert payload["names"][2] == "car"


def test_dataset_includes_test_split(tmp_path: Path) -> None:
    """A shipped test/ split is discovered and referenced in data.yaml."""
    for split, stem in (("train", "a"), ("valid", "b"), ("test", "c")):
        _write_split(tmp_path, split, stem)

    ds = ObbDataset.from_annotations_root(tmp_path)
    assert ds.count_pairs("test") == 1
    written = ds.write_ultralytics_data_yaml(tmp_path / "out.yaml")
    payload = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert payload["test"] == "test/images"


def test_dataset_accepts_val_directory_name(tmp_path: Path) -> None:
    """``val/`` is accepted as an alias for UAV-OBB's ``valid/``."""
    _write_split(tmp_path, "train", "a")
    _write_split(tmp_path, "val", "b")

    ds = ObbDataset.from_annotations_root(tmp_path)
    assert ds.count_pairs("valid") == 1
    written = ds.write_ultralytics_data_yaml(tmp_path / "out.yaml")
    payload = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert payload["val"] == "val/images"


def test_dataset_missing_train_raises(tmp_path: Path) -> None:
    """A tree without train/ fails clearly rather than yielding zero pairs."""
    _write_split(tmp_path, "valid", "b")
    with pytest.raises(FileNotFoundError, match="train/"):
        ObbDataset.from_annotations_root(tmp_path)


def test_dataset_missing_val_raises(tmp_path: Path) -> None:
    """A tree without valid/ or val/ fails clearly."""
    _write_split(tmp_path, "train", "a")
    with pytest.raises(FileNotFoundError, match="valid/"):
        ObbDataset.from_annotations_root(tmp_path)


def test_dataset_unknown_split_raises(tmp_path: Path) -> None:
    """Asking for a split that does not exist is an error, not an empty iterator."""
    _write_split(tmp_path, "train", "a")
    _write_split(tmp_path, "valid", "b")
    ds = ObbDataset.from_annotations_root(tmp_path)
    with pytest.raises(ValueError, match="Unknown or missing split"):
        list(ds.iter_pairs("nope"))


def test_iter_split_pairs_skips_unlabelled_images(tmp_path: Path) -> None:
    """Images with no matching label file are skipped."""
    _write_split(tmp_path, "train", "a")
    (tmp_path / "train" / "images" / "orphan.jpg").write_bytes(b"fake")
    assert len(list(iter_split_pairs(tmp_path / "train"))) == 1


def test_iter_split_pairs_requires_images_and_labels(tmp_path: Path) -> None:
    """A split directory missing images/ or labels/ fails clearly."""
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="images"):
        list(iter_split_pairs(tmp_path / "train"))

    (tmp_path / "train" / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="labels"):
        list(iter_split_pairs(tmp_path / "train"))
