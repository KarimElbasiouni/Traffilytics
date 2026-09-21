#!/usr/bin/env python3
"""Assign track_ids and write generated trajectories (Epic 3).

Thin CLI around :class:`computer_vision.tracking.tracker.VehicleTracker` and
:class:`computer_vision.trajectories.generator.TrajectoryGenerator`. Reads
``data/processed/<video_id>/detections.json`` (Epic 2) and writes
``trajectories.json``. Lane assignment is not applied yet (FR-TRK-006).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from computer_vision.detection.detector import DetectorError, load_detections_json
from computer_vision.preprocessing.config import load_config, resolve_data_path
from computer_vision.tracking.bytetrack import TrackerError
from computer_vision.tracking.tracker import VehicleTracker
from computer_vision.trajectories.generator import (
    DEFAULT_TRAJECTORIES_NAME,
    TrajectoryGenerator,
)


def _tracking_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read ``tracking.*`` from YAML, ignoring unknown keys."""
    return dict(cfg.get("tracking") or {})


def _print_plan(
    *,
    detections_path: Path,
    dest: Path,
    video_id: str | None,
    n_detections: int,
    tracker: str,
    min_hits: int,
) -> None:
    print("Detections:  ", detections_path)
    print("Video id:    ", video_id or "(none)")
    print("Detections n:", n_detections)
    print("Tracker:     ", tracker)
    print("Min hits:    ", min_hits)
    print("Output:      ", dest)


def main(argv: list[str] | None = None) -> int:
    """Load detections.json, run ByteTrack, write trajectories.json."""
    parser = argparse.ArgumentParser(
        description="Track vehicles in detections.json and write trajectories.json"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--video-id",
        default=None,
        help="Processed video id (reads data/processed/<id>/detections.json)",
    )
    parser.add_argument(
        "--detections",
        default=None,
        help="Explicit detections.json (overrides --video-id layout)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="trajectories.json path (default: next to detections.json)",
    )
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Override processed output root (default: data.processed from config)",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=None,
        help="Drop tracks shorter than this many frames (default from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate detections.json without running ByteTrack",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    track_cfg = _tracking_cfg(cfg)
    tracker_name = str(track_cfg.get("tracker") or "bytetrack")
    min_hits = (
        args.min_hits
        if args.min_hits is not None
        else int(track_cfg.get("min_hits") or 1)
    )

    processed_root = (
        Path(args.processed_root).expanduser()
        if args.processed_root
        else resolve_data_path(cfg, "processed")
    )

    if args.detections:
        detections_path = Path(args.detections).expanduser()
        if not detections_path.is_absolute():
            detections_path = (_REPO_ROOT / detections_path).resolve()
        else:
            detections_path = detections_path.resolve()
    elif args.video_id:
        detections_path = processed_root / args.video_id / "detections.json"
    else:
        print("ERROR: pass --video-id or --detections", file=sys.stderr)
        return 1

    dest = (
        Path(args.out).expanduser()
        if args.out
        else detections_path.parent / DEFAULT_TRAJECTORIES_NAME
    )
    if not dest.is_absolute():
        dest = (_REPO_ROOT / dest).resolve()

    try:
        detections, meta = load_detections_json(detections_path)
    except DetectorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    video_id = args.video_id or meta.get("video_id") or detections_path.parent.name
    _print_plan(
        detections_path=detections_path,
        dest=dest,
        video_id=str(video_id) if video_id else None,
        n_detections=len(detections),
        tracker=tracker_name,
        min_hits=min_hits,
    )

    if args.dry_run:
        print("Dry run OK — not starting tracking.")
        return 0

    try:
        tracker = VehicleTracker(track_cfg)
        tracked = tracker.track(detections)
    except TrackerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    generator = TrajectoryGenerator(min_hits=min_hits)
    trajectories = generator.generate(tracked, video_id=str(video_id))
    written = generator.write_json(
        trajectories,
        dest,
        video_id=str(video_id),
        tracker=tracker_name,
        extra={"detections": str(detections_path), "n_detections": len(detections)},
    )
    diag = json.loads(written.read_text(encoding="utf-8")).get("diagnostics") or {}
    print(f"Wrote {diag.get('n_tracks', 0)} tracks / {diag.get('n_points', 0)} points")
    print(f"Trajectories → {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
