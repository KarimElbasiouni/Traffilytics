"""Tests for DRIFT OBB annotation adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adapters.drift.obb_annotations import (
    DriftOBBDataset,
    parse_obb_label_line,
    load_obb_label_file,
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
