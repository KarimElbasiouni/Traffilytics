#!/usr/bin/env python3
"""Download UAV-OBB and install it into the Traffilytics data layout (Epic 1).

Splits and ``data.yaml`` land under ``data/annotations``; the bundled demo clips
land under ``data/raw`` so they can be ingested like any uploaded video.

The Mendeley public file listing needs no credentials, so a plain run downloads
the ~506 MiB archive and verifies the SHA-256 the API reports. Pass ``--src`` to
install from a zip or folder you already have, which skips the network entirely.
Extraction stages files next to the archive, so allow roughly 2.5 GB of free
space on the data volume for a full run.

The test suite never invokes this script (NFR-TEST-005).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adapters.uav_obb.acquire import (
    KAGGLE_MIRROR,
    MENDELEY_DOI,
    MENDELEY_LANDING_URL,
    RemoteFile,
    find_bundled_videos,
    find_dataset_root,
    install_splits,
    install_videos,
    mendeley_files_url,
    parse_file_listing,
    sha256_of,
)
from adapters.uav_obb.layout import DatasetLayout, repo_relative

_MANUAL_HINT = (
    f"Download UAV-OBB.zip manually from {MENDELEY_LANDING_URL} "
    f"(DOI {MENDELEY_DOI}) or the Kaggle mirror '{KAGGLE_MIRROR}', "
    "then re-run with --src /path/to/UAV-OBB.zip"
)


_USER_AGENT = "Traffilytics-dataset-fetcher/1.0 (+https://data.mendeley.com)"


class DownloadError(RuntimeError):
    """Any failure acquiring or unpacking the dataset."""


def _open(url: str, timeout: int):
    """Open an HTTPS URL with an explicit User-Agent; Mendeley rejects some defaults."""
    return urlopen(  # noqa: S310 - https endpoints from the Mendeley public API
        Request(url, headers={"User-Agent": _USER_AGENT}), timeout=timeout
    )


def _human(size: int) -> str:
    """Format a byte count as MiB/GiB for progress output."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GiB"


def _fetch_metadata() -> RemoteFile:
    """Ask Mendeley's public API which file to download and what it should hash to."""
    url = mendeley_files_url()
    try:
        with _open(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError) as exc:
        raise DownloadError(f"Could not reach Mendeley ({exc}). {_MANUAL_HINT}") from exc
    except json.JSONDecodeError as exc:
        raise DownloadError(f"Mendeley returned invalid JSON ({exc}). {_MANUAL_HINT}") from exc

    try:
        return parse_file_listing(payload)
    except ValueError as exc:
        raise DownloadError(f"{exc}. {_MANUAL_HINT}") from exc


def _download(remote: RemoteFile, dest: Path) -> None:
    """Stream the archive to ``dest`` via a .part file, printing coarse progress."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    next_report = 0
    try:
        with _open(remote.download_url, timeout=120) as response:
            total = int(response.headers.get("content-length") or remote.size or 0)
            with part.open("wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        pct = f" ({done * 100 // total}%)" if total else ""
                        print(f"  {_human(done)}{pct}", flush=True)
                        next_report = done + (64 << 20)
    except (URLError, TimeoutError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"Download failed ({exc}). {_MANUAL_HINT}") from exc
    part.replace(dest)
    print(f"  {_human(done)} (100%)")


def _verify(archive: Path, remote: RemoteFile) -> None:
    """Compare the downloaded archive against the checksum Mendeley published."""
    if not remote.sha256:
        print("WARNING: Mendeley reported no checksum; skipping verification")
        return
    print("Verifying SHA-256 ...")
    actual = sha256_of(archive)
    if actual != remote.sha256:
        raise DownloadError(
            f"Checksum mismatch for {archive.name}: expected {remote.sha256}, got {actual}. "
            "Delete the file and retry."
        )
    print("  checksum OK")


def _unzip(archive: Path, staging: Path) -> Path:
    """Extract a dataset archive into ``staging`` and return the staging path."""
    if not zipfile.is_zipfile(archive):
        raise DownloadError(f"Not a zip archive: {archive}")
    print(f"Extracting {archive.name} ...")
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DownloadError(f"Could not extract {archive}: {exc}") from exc
    return staging


def _resolve_source(args: argparse.Namespace, data_root: Path) -> tuple[Path, Path | None]:
    """Return (dataset tree to install from, staging dir to clean up afterwards)."""
    src = Path(args.src).expanduser()
    if not src.exists():
        raise DownloadError(f"--src not found: {src}")
    if src.is_dir():
        return find_dataset_root(src), None

    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".uav_obb_stage_", dir=data_root))
    try:
        return find_dataset_root(_unzip(src, staging)), staging
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _fetch_source(args: argparse.Namespace, data_root: Path) -> tuple[Path, Path | None]:
    """Download (or reuse) the archive, verify it, and stage its contents."""
    remote = _fetch_metadata()
    archive = Path(args.archive).expanduser() if args.archive else data_root / remote.filename
    print(f"Remote file: {remote.filename} ({_human(remote.size)})")
    print(f"Archive:     {repo_relative(archive)}")

    if archive.is_file() and not args.force:
        print("Archive already present; reusing it (pass --force to re-download)")
    else:
        print(f"Downloading from {remote.download_url} ...")
        _download(remote, archive)
    _verify(archive, remote)

    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".uav_obb_stage_", dir=data_root))
    try:
        dataset_root = find_dataset_root(_unzip(archive, staging))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"Removed {repo_relative(archive)} (pass --keep-archive to keep it)")
    return dataset_root, staging


def main(argv: list[str] | None = None) -> int:
    """Acquire UAV-OBB, install splits into data/annotations and clips into data/raw."""
    parser = argparse.ArgumentParser(
        description="Download UAV-OBB into the Traffilytics data layout"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--src",
        default=None,
        help="Install from a local UAV-OBB.zip or extracted folder instead of downloading",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Annotations root for the splits (default: data/annotations from config)",
    )
    parser.add_argument(
        "--raw-dest",
        default=None,
        help="Destination for bundled demo clips (default: data/raw from config)",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Where to write the download (default: <data root>/UAV-OBB.zip)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip after extracting",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Install only the annotated splits, not the bundled demo clips",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the archive is already present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be installed without downloading or copying",
    )
    args = parser.parse_args(argv)

    layout = DatasetLayout.from_config(args.config)
    dest = Path(args.dest).expanduser() if args.dest else layout.annotations
    raw_dest = Path(args.raw_dest).expanduser() if args.raw_dest else layout.raw

    staging: Path | None = None
    try:
        if args.dry_run and not args.src:
            remote = _fetch_metadata()
            print(f"Would download: {remote.filename} ({_human(remote.size)})")
            print(f"  url:      {remote.download_url}")
            print(f"  sha256:   {remote.sha256}")
            print(f"  splits  → {repo_relative(dest)}")
            print(f"  clips   → {repo_relative(raw_dest)}")
            return 0

        if args.src:
            dataset_root, staging = _resolve_source(args, layout.root)
        else:
            dataset_root, staging = _fetch_source(args, layout.root)

        videos = [] if args.skip_videos else find_bundled_videos(dataset_root)
        if args.dry_run:
            print(f"Would install from {dataset_root}")
            print(f"  splits → {repo_relative(dest)}")
            print(f"  clips  → {repo_relative(raw_dest)} ({len(videos)} file(s))")
            return 0

        copied = install_splits(dataset_root, dest)
        print(f"Installed into {repo_relative(dest)}: {', '.join(copied)}")
        if not args.skip_videos:
            written = install_videos(dataset_root, raw_dest)
            print(f"Copied {len(written)} demo clip(s) → {repo_relative(raw_dest)}")
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    print("\nNext: python scripts/prepare_obb_dataset.py  # writes the Ultralytics data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
