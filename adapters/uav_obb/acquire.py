"""Locate a downloaded UAV-OBB tree and install it into the Traffilytics data layout.

UAV-OBB is published on Mendeley Data as a single ``UAV-OBB.zip``. The public
file listing needs no credentials, so the download URL and its SHA-256 checksum
are discovered at runtime instead of being pinned here — only the dataset id and
version are fixed.

Everything in this module is filesystem-only and offline. The HTTP request lives
in ``scripts/download_uav_obb.py`` so the test suite never needs the network
(NFR-TEST-005).
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

MENDELEY_DATASET_ID = "6snrjwcpkh"
MENDELEY_DATASET_VERSION = 3
MENDELEY_DOI = "10.17632/6snrjwcpkh.3"
MENDELEY_LANDING_URL = f"https://data.mendeley.com/datasets/{MENDELEY_DATASET_ID}/3"
KAGGLE_MIRROR = "mdferozahmedafm/uav-obb-drone-based-urban-vehicle-dataset"

# UAV-OBB ships `valid`; `val` is accepted for hand-assembled copies.
SPLIT_DIR_NAMES = ("train", "valid", "val", "test")
DATA_YAML_NAME = "data.yaml"
_VIDEO_DIR_NAME = "test_videos_mp4"
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def mendeley_files_url(version: int = MENDELEY_DATASET_VERSION) -> str:
    """Public (no-auth) file-listing endpoint for the dataset."""
    return (
        "https://data.mendeley.com/public-api/datasets/"
        f"{MENDELEY_DATASET_ID}/files?folder_id=root&version={version}"
    )


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable dataset file as described by the Mendeley listing."""

    filename: str
    size: int
    sha256: str
    download_url: str


def parse_file_listing(payload: Any, *, suffix: str = ".zip") -> RemoteFile:
    """Pick the dataset archive out of a Mendeley public file listing.

    Raises ValueError when the payload is not a list of entries or holds no
    matching file with a download URL.
    """
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise ValueError("Expected a list of file entries from the Mendeley API")

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("filename") or "")
        if not name.lower().endswith(suffix):
            continue
        details = entry.get("content_details") or {}
        url = details.get("download_url")
        if not url:
            continue
        return RemoteFile(
            filename=name,
            size=int(entry.get("size") or details.get("size") or 0),
            sha256=str(details.get("sha256_hash") or ""),
            download_url=str(url),
        )
    raise ValueError(f"No '*{suffix}' file with a download URL in the Mendeley listing")


def sha256_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_splits(path: Path) -> bool:
    """True when a directory holds a ``train`` split plus ``valid`` or ``val``."""
    if not (path / "train").is_dir():
        return False
    return (path / "valid").is_dir() or (path / "val").is_dir()


def find_dataset_root(extracted: str | Path) -> Path:
    """Return the directory containing the split folders.

    ``UAV-OBB.zip`` nests everything under a ``UAV-OBB/`` folder, but a
    hand-extracted copy may put the splits at the top level, so both shapes are
    accepted.
    """
    root = Path(extracted).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    if has_splits(root):
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if has_splits(child):
            return child
    raise FileNotFoundError(
        f"No train/ + valid/ split tree in {root} or its immediate subfolders"
    )


def install_splits(dataset_root: str | Path, dest: str | Path) -> list[str]:
    """Copy split folders and ``data.yaml`` into ``dest``; return what was copied."""
    source = Path(dataset_root).expanduser().resolve()
    target = Path(dest).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in SPLIT_DIR_NAMES:
        split_dir = source / name
        if split_dir.is_dir():
            shutil.copytree(split_dir, target / name, dirs_exist_ok=True)
            copied.append(name)
    if not copied:
        raise FileNotFoundError(f"No split folders found under {source}")

    src_yaml = source / DATA_YAML_NAME
    if src_yaml.is_file():
        shutil.copy2(src_yaml, target / DATA_YAML_NAME)
        copied.append(DATA_YAML_NAME)
    return copied


def find_bundled_videos(dataset_root: str | Path) -> list[Path]:
    """Locate the supplementary demo clips shipped alongside the splits."""
    source = Path(dataset_root).expanduser().resolve()
    found: list[Path] = []
    for child in sorted(source.iterdir()):
        if child.is_dir() and child.name.lower() == _VIDEO_DIR_NAME:
            found.extend(
                sorted(p for p in child.rglob("*") if p.suffix.lower() in _VIDEO_SUFFIXES)
            )
        elif child.is_file() and child.suffix.lower() in _VIDEO_SUFFIXES:
            found.append(child)
    return found


def install_videos(dataset_root: str | Path, raw_dest: str | Path) -> list[Path]:
    """Copy bundled demo clips into ``data/raw`` and return their new paths."""
    videos = find_bundled_videos(dataset_root)
    if not videos:
        return []
    target = Path(raw_dest).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy2(video, target / video.name)) for video in videos]
