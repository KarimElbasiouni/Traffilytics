"""Tests for the UAV-OBB path layout helpers (no network / dataset required)."""

from __future__ import annotations

from pathlib import Path

from adapters.uav_obb.layout import (
    DatasetLayout,
    is_stabilized_name,
    list_raw_videos,
    prefer_stabilized,
    repo_relative,
)


def _layout(tmp_path: Path, *, raw: Path | None = None) -> DatasetLayout:
    """Build a DatasetLayout rooted at tmp_path without touching the real config."""
    return DatasetLayout(
        root=tmp_path,
        raw=raw if raw is not None else tmp_path,
        annotations=tmp_path,
        processed=tmp_path,
    )


def test_prefer_stabilized_ordering(tmp_path: Path) -> None:
    """prefer_stabilized sorts clips with ``stabil`` in the name ahead of others."""
    a = tmp_path / "clip_01_raw.mp4"
    b = tmp_path / "clip_01_stabilized.mp4"
    a.write_bytes(b"")
    b.write_bytes(b"")

    ordered = prefer_stabilized([a, b])
    assert ordered[0].name == "clip_01_stabilized.mp4"
    assert is_stabilized_name(ordered[0].name)
    assert not is_stabilized_name(ordered[1].name)


def test_layout_from_config() -> None:
    """DatasetLayout.from_config resolves expected data folders and can create them."""
    layout = DatasetLayout.from_config()
    assert layout.raw.name == "raw"
    assert layout.annotations.name == "annotations"
    assert layout.processed.name == "processed"
    layout.ensure_dirs()
    assert layout.raw.is_dir()


def test_list_raw_videos_filters_by_extension(tmp_path: Path) -> None:
    """Only known video extensions are listed, sorted by name."""
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "a.mov").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    assert [p.name for p in list_raw_videos(_layout(tmp_path))] == ["a.mov", "b.mp4"]


def test_list_raw_videos_missing_dir(tmp_path: Path) -> None:
    """A missing raw/ folder yields an empty list rather than raising."""
    assert list_raw_videos(_layout(tmp_path, raw=tmp_path / "absent")) == []


def test_repo_relative_outside_repo(tmp_path: Path) -> None:
    """A path outside the repo falls back to its absolute form."""
    assert Path(repo_relative(tmp_path)).is_absolute()
