#!/usr/bin/env python3
"""Download a small DRIFT sample into Traffilytics data/ layout.

Videos are not hosted on the Hugging Face trajectory dataset. This script pulls:
  - A small demo MP4 from the DRIFT GitHub repo → data/raw/
  - One GT trajectory CSV from Hugging Face (eval only) → data/gt_trajectories/
  - Optional YOLO OBB label/image samples from GitHub → data/annotations/

Those annotation samples are **two frames** for layout smoke only. The full
GitHub ``model/{train,valid,test}`` tree (~2,301 frames / ~300K instances) is
gitignored and must not be fetched in CI — use ``scripts/download_obb_dataset.py``.

Prefer stabilized clips when available; Traffilytics does not reimplement Stabilo.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from adapters.drift.drift_layout import DriftLayout, HF_DATASET_ID

# ~3.4 MB demo clip from DRIFT GitHub (good smoke-ingest target)
GITHUB_SAMPLE_VIDEO_URL = (
    "https://raw.githubusercontent.com/AIxMobility/The-DRIFT/main/od_small.mp4"
)
GITHUB_SAMPLE_VIDEO_NAME = "site_A_od_small.mp4"

# Smallish GT CSV on HF for layout smoke (eval only — not live trajectories)
HF_GT_SAMPLE = "A/drone_12.csv"

# Tiny YOLO OBB samples from DRIFT GitHub for annotation layout
GITHUB_ANN_FILES = (
    ("model/train/images/A_frame_0000.jpg", "train/images/A_frame_0000.jpg"),
    ("model/train/labels/A_frame_0000.txt", "train/labels/A_frame_0000.txt"),
    ("model/valid/images/A_frame_0001.jpg", "valid/images/A_frame_0001.jpg"),
    ("model/valid/labels/A_frame_0001.txt", "valid/labels/A_frame_0001.txt"),
    ("model/data.yaml", "data.yaml"),
)


def _download_url(url: str, dest: Path, *, token: str | None = None) -> None:
    """Download a URL to ``dest`` in chunks; optionally send an HF Bearer token."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "traffilytics-download/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} downloading {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error downloading {url}: {exc}") from exc


def download_sample_video(layout: DriftLayout, *, force: bool = False) -> Path:
    """Fetch the small GitHub demo MP4 into ``data/raw/`` (skips if already present)."""
    dest = layout.raw / GITHUB_SAMPLE_VIDEO_NAME
    if dest.is_file() and not force:
        print(f"Video already present: {dest}")
        return dest
    print(f"Downloading sample video → {dest}")
    _download_url(GITHUB_SAMPLE_VIDEO_URL, dest)
    print(f"Saved {dest.stat().st_size} bytes")
    return dest


def download_gt_sample(layout: DriftLayout, *, force: bool = False, token: str | None = None) -> Path:
    """Download one HF GT trajectory CSV into data/gt_trajectories/site_A/."""
    dest = layout.gt_trajectories / "site_A" / Path(HF_GT_SAMPLE).name
    if dest.is_file() and not force:
        print(f"GT CSV already present: {dest}")
        return dest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for GT download. "
            "Install deps (pip install -e .) or skip with --skip-gt."
        ) from exc

    print(f"Downloading HF GT sample {HF_DATASET_ID}/{HF_GT_SAMPLE} …")
    cached = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_GT_SAMPLE,
        repo_type="dataset",
        token=token,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = Path(cached).read_bytes()
    dest.write_bytes(data)
    print(f"Saved GT CSV → {dest} ({len(data)} bytes) [eval only]")
    return dest


def download_annotation_samples(layout: DriftLayout, *, force: bool = False) -> list[Path]:
    """Copy a few GitHub YOLO image/label samples into ``data/annotations/`` for layout smoke tests."""
    base = "https://raw.githubusercontent.com/AIxMobility/The-DRIFT/main/"
    written: list[Path] = []
    for remote, local_rel in GITHUB_ANN_FILES:
        dest = layout.annotations / local_rel
        if dest.is_file() and not force:
            print(f"Annotation already present: {dest}")
            written.append(dest)
            continue
        print(f"Downloading annotation sample → {dest}")
        _download_url(base + remote, dest)
        written.append(dest)
    return written


def build_parser() -> argparse.ArgumentParser:
    """Define CLI flags for which sample assets to download and whether to ingest afterward."""
    p = argparse.ArgumentParser(
        description="Download DRIFT sample assets into data/",
        epilog=(
            "Full OBB splits (~2,301 frames / ~300K instances, gitignored): "
            "python scripts/download_obb_dataset.py --full --src /path/to/The-DRIFT"
        ),
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-gt", action="store_true", help="Skip Hugging Face GT CSV")
    p.add_argument("--skip-annotations", action="store_true")
    p.add_argument(
        "--ingest",
        action="store_true",
        help="After download, run Traffilytics ingest on the sample video",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Download selected sample assets into data/, optionally run ingest, return exit code."""
    load_dotenv(_REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    token = os.environ.get("HF_TOKEN") or None

    layout = DriftLayout.from_config(args.config)
    layout.ensure_dirs()

    video_path: Path | None = None
    try:
        if not args.skip_video:
            video_path = download_sample_video(layout, force=args.force)
        if not args.skip_gt:
            download_gt_sample(layout, force=args.force, token=token)
        if not args.skip_annotations:
            download_annotation_samples(layout, force=args.force)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nLayout ready:")
    print(f"  raw:            {layout.raw}")
    print(f"  annotations:    {layout.annotations}")
    print(f"  gt_trajectories:{layout.gt_trajectories}  (eval only)")
    print(f"  processed:      {layout.processed}")

    if args.ingest:
        if video_path is None:
            video_path = layout.raw / GITHUB_SAMPLE_VIDEO_NAME
        if not video_path.is_file():
            print("ERROR: No sample video to ingest.", file=sys.stderr)
            return 1
        import importlib.util

        ingest_path = _REPO_ROOT / "scripts" / "ingest_video.py"
        spec = importlib.util.spec_from_file_location("ingest_video", ingest_path)
        if spec is None or spec.loader is None:
            print("ERROR: Unable to load ingest_video.py", file=sys.stderr)
            return 1
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(
            [
                "--video",
                str(video_path),
                "--config",
                args.config,
                "--max-frames",
                "30",
                "--stride",
                "5",
            ]
        )

    print("\nNext: python scripts/ingest_video.py --video data/raw/site_A_od_small.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
