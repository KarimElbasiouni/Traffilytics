"""Tests for generic YOLO-OBB label parsing (core, dataset-agnostic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from computer_vision.detection.obb_labels import (
    OBBBox,
    load_obb_label_file,
    parse_obb_label_line,
)


def test_parse_obb_label_line() -> None:
    """A valid 9-field OBB line becomes an OBBBox with class car and 8 corners."""
    box = parse_obb_label_line("2 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4")
    assert box.class_id == 2
    assert box.class_name == "car"
    assert len(box.corners) == 8


def test_parse_obb_label_line_rejects_bad() -> None:
    """Malformed label lines (wrong field count) raise ValueError."""
    with pytest.raises(ValueError):
        parse_obb_label_line("1 0.1 0.2")


def test_parse_obb_label_line_accepts_float_class_id() -> None:
    """Ultralytics exports sometimes write the class id as a float."""
    box = parse_obb_label_line("5.0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4")
    assert box.class_id == 5
    assert box.class_name == "truck"


def test_obb_box_unknown_class_stringifies() -> None:
    """An id outside the class map falls back to its string form."""
    assert OBBBox(class_id=9, corners=(0.0,) * 8).class_name == "9"


def test_load_obb_label_file_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines are ignored; every other line becomes a box."""
    path = tmp_path / "frame.txt"
    path.write_text(
        "1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n"
        "\n"
        "5 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n",
        encoding="utf-8",
    )
    boxes = load_obb_label_file(path)
    assert [b.class_id for b in boxes] == [1, 5]
    assert [b.class_name for b in boxes] == ["bus", "truck"]


def test_load_obb_label_file_missing(tmp_path: Path) -> None:
    """A missing label file fails clearly (NFR-ACC-004)."""
    with pytest.raises(FileNotFoundError):
        load_obb_label_file(tmp_path / "no_such_label.txt")
