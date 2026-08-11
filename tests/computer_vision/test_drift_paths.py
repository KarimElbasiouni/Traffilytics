"""Tests for DRIFT path adapter (no network / dataset required)."""

from __future__ import annotations

from pathlib import Path

from adapters.drift.paths import (
    DriftLayout,
    is_stabilized_name,
    parse_site_from_name,
    prefer_stabilized,
)


def test_parse_site_from_name() -> None:
    """Numeric and letter site ids are parsed from common DRIFT-style filenames."""
    assert parse_site_from_name("site_03_clip_01.mp4") == "03"
    assert parse_site_from_name("site-07-stabilized") == "07"
    assert parse_site_from_name("site_A_od_small.mp4") == "A"
    assert parse_site_from_name("A_frame_0000.jpg") == "A"
    assert parse_site_from_name("random_clip.mp4") is None


def test_prefer_stabilized_ordering(tmp_path: Path) -> None:
    """prefer_stabilized sorts clips with ``stabil`` in the name ahead of others."""
    a = tmp_path / "site_01_raw.mp4"
    b = tmp_path / "site_01_stabilized.mp4"
    a.write_bytes(b"")
    b.write_bytes(b"")
    ordered = prefer_stabilized([a, b])
    assert ordered[0].name == "site_01_stabilized.mp4"
    assert is_stabilized_name(ordered[0].name)


def test_layout_from_config(tmp_path: Path, monkeypatch) -> None:
    """DriftLayout.from_config resolves expected data folders and can create them."""
    # Use real default config; ensure dirs resolve under repo
    layout = DriftLayout.from_config()
    assert layout.raw.name == "raw"
    assert layout.annotations.name == "annotations"
    assert layout.gt_trajectories.name == "gt_trajectories"
    assert "Hj-Lee/The-DRIFT" in layout.hf_dataset
    layout.ensure_dirs()
    assert layout.raw.is_dir()
