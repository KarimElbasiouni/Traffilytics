"""UAV-OBB dataset path conventions — isolated from core VideoProcessor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from computer_vision.preprocessing.config import REPO_ROOT, load_config, resolve_data_path


@dataclass(frozen=True)
class DatasetLayout:
    """Local directory layout for dataset splits, input video, and processed artifacts."""

    root: Path
    raw: Path
    annotations: Path
    processed: Path

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> DatasetLayout:
        """Build a DatasetLayout from ``configs/default.yaml`` (or another YAML path)."""
        cfg = load_config(config_path)
        return cls(
            root=resolve_data_path(cfg, "root"),
            raw=resolve_data_path(cfg, "raw"),
            annotations=resolve_data_path(cfg, "annotations"),
            processed=resolve_data_path(cfg, "processed"),
        )

    def ensure_dirs(self) -> None:
        """Create raw, annotations, and processed folders if missing."""
        for path in (self.raw, self.annotations, self.processed):
            path.mkdir(parents=True, exist_ok=True)


def is_stabilized_name(name: str) -> bool:
    """Return True if the filename looks like a pre-stabilized clip (contains ``stabil``)."""
    return "stabil" in Path(name).name.lower()


def list_raw_videos(
    layout: DatasetLayout | None = None,
    *,
    extensions: Iterable[str] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[Path]:
    """List video files under ``data/raw``."""
    layout = layout or DatasetLayout.from_config()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    if not layout.raw.is_dir():
        return []
    return sorted(
        p for p in layout.raw.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def prefer_stabilized(paths: Iterable[Path]) -> list[Path]:
    """Sort paths so pre-stabilized clips come first; stabilization is not implemented here."""
    items = list(paths)
    return sorted(items, key=lambda p: (0 if is_stabilized_name(p.name) else 1, p.name))


def repo_relative(path: Path) -> str:
    """Format a path relative to the repo root when possible; otherwise absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())
