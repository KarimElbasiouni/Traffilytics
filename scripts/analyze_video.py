#!/usr/bin/env python3
"""Run Epic 4 analytics on generated trajectories.

Thin CLI around :class:`analytics.engine.AnalyticsEngine`. Reads
``data/processed/<video_id>/trajectories.json`` and writes ``analytics.json``.
Optional ``--lanes`` supplies zone polygons for bottlenecks; optional
``--pixels-per-metre`` converts speed/density to physical units.
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

from analytics.engine import AnalyticsEngine
from analytics.types import DEFAULT_ANALYTICS_NAME, AnalyticsError, SceneContext
from computer_vision.preprocessing.config import load_config, resolve_data_path
from computer_vision.trajectories.generator import (
    DEFAULT_TRAJECTORIES_NAME,
    TrajectoryError,
    load_trajectories_json,
)
from computer_vision.trajectories.lanes import LaneAssigner, LaneConfigError


def _analytics_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("analytics") or {})


def _parse_resolution(raw: Any) -> tuple[int | None, int | None]:
    if not raw:
        return None, None
    text = str(raw).lower().replace(" ", "")
    if "x" not in text:
        return None, None
    left, right = text.split("x", 1)
    try:
        return int(float(left)), int(float(right))
    except ValueError:
        return None, None


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _print_plan(
    *,
    trajectories_path: Path,
    dest: Path,
    video_id: str,
    n_tracks: int,
    fps: float,
    pixels_per_metre: float | None,
    lanes_path: Path | None,
    n_zones: int,
) -> None:
    print("Trajectories:", trajectories_path)
    print("Video id:    ", video_id)
    print("Tracks n:    ", n_tracks)
    print("fps:         ", fps)
    print("Scale:       ", pixels_per_metre if pixels_per_metre else "(pixel units)")
    print("Lanes/zones: ", lanes_path or "(none)")
    print("Zones n:     ", n_zones)
    print("Output:      ", dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze generated trajectories (flow, bottleneck, imbalance, events, insights)"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--video-id",
        default=None,
        help="Processed video id (reads data/processed/<id>/trajectories.json)",
    )
    parser.add_argument(
        "--trajectories",
        default=None,
        help="Explicit trajectories.json (overrides --video-id layout)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="analytics.json path (default: next to trajectories.json)",
    )
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Override processed output root (default: data.processed from config)",
    )
    parser.add_argument(
        "--lanes",
        default=None,
        help="Lane/zone JSON (default: configs/lanes/<video_id>.json if present)",
    )
    parser.add_argument(
        "--pixels-per-metre",
        type=float,
        default=None,
        help="Optional scale; omit to keep pixel-based speed/density labels",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override fps (default: metadata.json next to trajectories)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate trajectories.json without writing analytics",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    an_cfg = _analytics_cfg(cfg)
    processed_root = (
        Path(args.processed_root).expanduser()
        if args.processed_root
        else resolve_data_path(cfg, "processed")
    )

    if args.trajectories:
        traj_path = Path(args.trajectories).expanduser()
        if not traj_path.is_absolute():
            traj_path = (_REPO_ROOT / traj_path).resolve()
        else:
            traj_path = traj_path.resolve()
    elif args.video_id:
        traj_path = processed_root / args.video_id / DEFAULT_TRAJECTORIES_NAME
    else:
        print("ERROR: pass --video-id or --trajectories", file=sys.stderr)
        return 1

    dest = (
        Path(args.out).expanduser()
        if args.out
        else traj_path.parent / DEFAULT_ANALYTICS_NAME
    )
    if not dest.is_absolute():
        dest = (_REPO_ROOT / dest).resolve()

    try:
        trajectories, meta = load_trajectories_json(traj_path)
    except TrajectoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    video_id = str(args.video_id or meta.get("video_id") or traj_path.parent.name)
    metadata = _load_metadata(traj_path.parent / "metadata.json")
    fps = float(args.fps or metadata.get("fps") or 0.0)
    width, height = _parse_resolution(metadata.get("resolution"))
    scale = args.pixels_per_metre
    if scale is None:
        raw_scale = an_cfg.get("pixels_per_metre")
        scale = float(raw_scale) if raw_scale not in (None, "") else None

    try:
        assigner = _resolve_lanes(explicit=args.lanes, video_id=video_id, cfg=cfg)
    except LaneConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    n_zones = len(assigner.zones) if assigner is not None else 0
    _print_plan(
        trajectories_path=traj_path,
        dest=dest,
        video_id=video_id,
        n_tracks=len(trajectories),
        fps=fps,
        pixels_per_metre=scale,
        lanes_path=assigner.source if assigner is not None else None,
        n_zones=n_zones,
    )

    if args.dry_run:
        print("Dry run OK — not starting analytics.")
        return 0

    if fps <= 0:
        print(
            "ERROR: fps is missing or zero. Pass --fps or ingest metadata.json.",
            file=sys.stderr,
        )
        return 1

    context = SceneContext(
        video_id=video_id,
        fps=fps,
        width=width,
        height=height,
        pixels_per_metre=scale,
        assigner=assigner,
        source=str(traj_path),
        tracker=str(meta.get("tracker")) if meta.get("tracker") else None,
    )
    try:
        engine = AnalyticsEngine(an_cfg)
        report = engine.analyze(trajectories, context=context)
        written = engine.write_json(report, dest)
    except AnalyticsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    print(f"Mean speed:  {report.flow.mean_speed} {report.units.speed_unit}")
    print(f"Events:      {len(report.events)}")
    print(f"Analytics  → {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
