#!/usr/bin/env python3
"""Assign track_ids and write generated trajectories (Epic 3).

Thin CLI around :class:`computer_vision.tracking.tracker.VehicleTracker` and
:class:`computer_vision.trajectories.generator.TrajectoryGenerator`. Reads
``data/processed/<video_id>/detections.json`` (Epic 2) and writes
``trajectories.json``. Optional ``--overlays`` draws track_id on ingested
frames; optional ``--lanes`` stamps FR-TRK-006 lane labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from computer_vision.detection.detector import DetectorError, load_detections_json
from computer_vision.preprocessing.config import load_config, resolve_data_path
from computer_vision.tracking.bytetrack import TrackerError
from computer_vision.tracking.diagnostics import (
    DEFAULT_DIAGNOSTICS_NAME,
    TrackingDiagnostics,
)
from computer_vision.tracking.overlay import (
    DEFAULT_OVERLAY_DIRNAME,
    DEFAULT_OVERLAY_VIDEO_NAME,
    boxes_for_overlay,
    write_overlay_stills,
    write_overlay_video,
)
from computer_vision.tracking.tracker import VehicleTracker
from computer_vision.trajectories.generator import (
    DEFAULT_TRAJECTORIES_NAME,
    TrajectoryGenerator,
)
from computer_vision.trajectories.lanes import LaneAssigner, LaneConfigError


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
    lanes_path: Path | None,
    overlay_dir: Path | None,
    overlay_video: Path | None,
) -> None:
    print("Detections:  ", detections_path)
    print("Video id:    ", video_id or "(none)")
    print("Detections n:", n_detections)
    print("Tracker:     ", tracker)
    print("Min hits:    ", min_hits)
    print("Output:      ", dest)
    print("Lanes:       ", lanes_path or "(none)")
    if overlay_dir is not None:
        print("Overlays:    ", overlay_dir)
    if overlay_video is not None:
        print("Overlay mp4: ", overlay_video)


def _resolve_lanes(
    *,
    explicit: str | None,
    video_id: str,
    cfg: dict[str, Any],
) -> LaneAssigner | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        return LaneAssigner.from_path(path)
    lanes_cfg = cfg.get("lanes") or {}
    lanes_dir = lanes_cfg.get("dir") or "configs/lanes"
    return LaneAssigner.discover(video_id, lanes_dir=lanes_dir, repo_root=_REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    """Load detections.json, run ByteTrack, write trajectories and diagnostics."""
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
        "--lanes",
        default=None,
        help="Lane/zone JSON (default: configs/lanes/<video_id>.json if present)",
    )
    parser.add_argument(
        "--overlays",
        action="store_true",
        help="Write track_id overlay stills and MP4 from ingested frames",
    )
    parser.add_argument(
        "--overlay-dir",
        default=None,
        help="Override overlay stills directory (implies --overlays)",
    )
    parser.add_argument(
        "--overlay-video",
        default=None,
        help="Override overlay MP4 path (implies --overlays)",
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
    want_overlays = bool(args.overlays or args.overlay_dir or args.overlay_video)

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

    video_id = str(args.video_id or meta.get("video_id") or detections_path.parent.name)

    overlay_dir = None
    overlay_video = None
    if want_overlays:
        overlay_dir = (
            Path(args.overlay_dir).expanduser()
            if args.overlay_dir
            else detections_path.parent / DEFAULT_OVERLAY_DIRNAME
        )
        if not overlay_dir.is_absolute():
            overlay_dir = (_REPO_ROOT / overlay_dir).resolve()
        overlay_video = (
            Path(args.overlay_video).expanduser()
            if args.overlay_video
            else detections_path.parent / DEFAULT_OVERLAY_VIDEO_NAME
        )
        if not overlay_video.is_absolute():
            overlay_video = (_REPO_ROOT / overlay_video).resolve()

    try:
        assigner = _resolve_lanes(explicit=args.lanes, video_id=video_id, cfg=cfg)
    except LaneConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    _print_plan(
        detections_path=detections_path,
        dest=dest,
        video_id=video_id,
        n_detections=len(detections),
        tracker=tracker_name,
        min_hits=min_hits,
        lanes_path=assigner.source if assigner is not None else None,
        overlay_dir=overlay_dir,
        overlay_video=overlay_video,
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
    trajectories = generator.generate(
        tracked, video_id=video_id, assigner=assigner
    )
    diag = TrackingDiagnostics(
        id_switch_max_gap=int(track_cfg.get("id_switch_max_gap") or 5),
        id_switch_max_dist=float(track_cfg.get("id_switch_max_dist") or 80),
    )
    report = diag.report(trajectories)
    extra: dict[str, Any] = {
        "detections": str(detections_path),
        "n_detections": len(detections),
        "lane_config": str(assigner.source) if assigner and assigner.source else None,
    }
    written = generator.write_json(
        trajectories,
        dest,
        video_id=video_id,
        tracker=tracker_name,
        extra={**extra, "diagnostics": report},
    )
    diag_path = diag.write_json(
        trajectories,
        dest.parent / DEFAULT_DIAGNOSTICS_NAME,
        extra={"video_id": video_id, "tracker": tracker_name},
    )
    print(f"Wrote {report.get('n_tracks', 0)} tracks / {report.get('n_points', 0)} points")
    print(f"Suspected ID switches: {report.get('suspected_id_switches', 0)}")
    print(f"Trajectories → {written}")
    print(f"Diagnostics  → {diag_path}")

    if want_overlays:
        frames_dir = detections_path.parent / "frames"
        fps = float(track_cfg.get("overlay_fps") or 10)
        try:
            labelled, extra = boxes_for_overlay(trajectories, detections)
            stills = write_overlay_stills(
                frames_dir,
                labelled,
                overlay_dir or dest.parent,
                extra_detections=extra,
            )
            video_path = write_overlay_video(
                frames_dir,
                labelled,
                overlay_video or dest.parent / DEFAULT_OVERLAY_VIDEO_NAME,
                extra_detections=extra,
                fps=fps,
            )
        except DetectorError as exc:
            print(f"WARNING: overlays skipped ({exc})", file=sys.stderr)
        else:
            print(f"Overlays:     {len(stills)} stills → {overlay_dir}")
            print(f"Overlay mp4 → {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
