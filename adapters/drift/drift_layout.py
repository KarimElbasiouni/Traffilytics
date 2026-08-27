"""DRIFT dataset path conventions — isolated from core VideoProcessor."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from computer_vision.preprocessing.config import REPO_ROOT, load_config, resolve_data_path

HF_DATASET_ID = "Hj-Lee/The-DRIFT"

_SITE_RE = re.compile(r"site[_\-]?(\d+|[A-I])", re.IGNORECASE)
_LETTER_SITE_RE = re.compile(r"(?:^|[_\-])([A-I])(?:[_\-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class DriftLayout:
    """Local directory layout for DRIFT media and eval artifacts."""

    root: Path
    raw: Path
    annotations: Path
    gt_trajectories: Path
    processed: Path
    hf_dataset: str = HF_DATASET_ID

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> DriftLayout:
        """Build a DriftLayout from ``configs/default.yaml`` (or another YAML path)."""
        cfg = load_config(config_path)
        data_root = resolve_data_path(cfg, "root")
        drift_cfg = cfg.get("drift") or {}
        hf = drift_cfg.get("hf_dataset") or HF_DATASET_ID
        return cls(
            root=data_root,
            raw=resolve_data_path(cfg, "raw"),
            annotations=resolve_data_path(cfg, "annotations"),
            gt_trajectories=resolve_data_path(cfg, "gt_trajectories"),
            processed=resolve_data_path(cfg, "processed"),
            hf_dataset=hf,
        )

    def ensure_dirs(self) -> None:
        """Create raw, annotations, gt_trajectories, and processed folders if missing."""
        for path in (self.raw, self.annotations, self.gt_trajectories, self.processed):
            path.mkdir(parents=True, exist_ok=True)


def parse_site_from_name(name: str) -> str | None:
    """Extract DRIFT site id from a filename or stem (e.g. site_03_clip → 03, site_A → A)."""
    stem = Path(name).stem
    match = _SITE_RE.search(stem)
    if match:
        value = match.group(1)
        return value.upper() if value.isalpha() else value
    match = _LETTER_SITE_RE.search(stem)
    if match:
        return match.group(1).upper()
    return None


def is_stabilized_name(name: str) -> bool:
    """Return True if the filename looks like a pre-stabilized clip (contains ``stabil``)."""
    return "stabil" in Path(name).name.lower()


def list_raw_videos(
    layout: DriftLayout | None = None,
    *,
    extensions: Iterable[str] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[Path]:
    """List video files under ``data/raw``."""
    layout = layout or DriftLayout.from_config()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    if not layout.raw.is_dir():
        return []
    return sorted(
        p for p in layout.raw.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def prefer_stabilized(paths: Iterable[Path]) -> list[Path]:
    """Sort paths so stabilized clips come first (policy, not Stabilo R&D)."""
    items = list(paths)
    return sorted(items, key=lambda p: (0 if is_stabilized_name(p.name) else 1, p.name))


def annotation_dir_for_site(layout: DriftLayout, site: str) -> Path:
    """Conventional annotations path for a site (may not exist yet)."""
    return layout.annotations / f"site_{site}"


def gt_trajectory_dir_for_site(layout: DriftLayout, site: str) -> Path:
    """Conventional GT trajectory CSV path for a site (eval only)."""
    return layout.gt_trajectories / f"site_{site}"


def repo_relative(path: Path) -> str:
    """Format a path relative to the repo root when possible; otherwise absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())
