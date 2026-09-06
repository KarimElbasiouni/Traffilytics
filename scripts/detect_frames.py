#!/usr/bin/env python3
"""Run Traffilytics OBB detection on ingested frames or a video (Epic 2).

Thin CLI around :class:`computer_vision.detection.detector.VehicleDetector`.
Writes ``data/processed/<video_id>/detections.json`` (FR-DET records) so Epic 3
can consume a stable file contract. Optional ``--overlays`` writes
``data/processed/<video_id>/det_overlays/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from computer_vision.detection.detector import (
    DEFAULT_DETECTIONS_NAME,
    DEFAULT_OVERLAY_DIRNAME,
    DetectWriteResult,
    DetectorError,
    VehicleDetector,
    detect_and_write,
    detect_video,
    list_frame_images,
)
from computer_vision.preprocessing.config import load_config, resolve_data_path


def _detection_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read ``detection.*`` (falling back to ``models.weights``) from YAML config."""
    det = cfg.get("detection") or {}
    models = cfg.get("models") or {}
    return {
        "weights": det.get("weights") or models.get("weights") or "models/your_obb.pt",
        "conf_threshold": float(det.get("conf_threshold", 0.25)),
        "iou_threshold": float(det.get("iou_threshold", 0.7)),
        "imgsz": int(det.get("imgsz", 640)),
        "device": str(det.get("device") or "auto"),
        "allow_pretrained": bool(det.get("allow_pretrained", False)),
    }


def _resolve_user_path(value: str, repo_root: Path) -> Path:
    """Resolve a user path against cwd first, then the repo root."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (repo_root / path).resolve()


def _resolve_weights_path(weights: str, repo_root: Path) -> Path:
    """Resolve product-weight paths relative to the repo (not cwd)."""
    path = Path(weights)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def _print_dry_run(
    *,
    weights: Path,
    weights_missing: bool,
    allow_pretrained: bool,
    source_kind: str | None,
    source_path: Path | None,
    n_frames: int | None,
    dest: Path | None,
    overlay_dir: Path | None,
    video_id: str | None,
    device: str,
) -> None:
    """Print resolved detect paths without running inference."""
    print("Weights:     ", weights)
    if weights_missing:
        hint = (
            "would use yolo11n-obb.pt (not FR-DET-001)"
            if allow_pretrained
            else "real run needs your_obb.pt or --allow-pretrained"
        )
        print(f"              (missing — {hint})")
    print("Device:      ", device)
    print("Video id:    ", video_id or "(none)")
    if source_kind and source_path is not None:
        extra = f" ({n_frames} images)" if n_frames is not None else ""
        print(f"Source:       {source_kind} {source_path}{extra}")
    else:
        print("Source:       (none — pass --video-id, --frames, or --video)")
    print("Output:      ", dest or "(none)")
    if overlay_dir is not None:
        print("Overlays:    ", overlay_dir)
    print("Dry run OK — not running inference.")


def _print_result(result: DetectWriteResult) -> None:
    """Print written detections.json path and counts after a real run."""
    print(f"Wrote {result.n_frames} frames / {len(result.detections)} detections")
    print(f"Detections → {result.detections_path}")
    if result.overlay_paths:
        overlay_dir = result.overlay_paths[0].parent
        print(f"Overlays:     {len(result.overlay_paths)} → {overlay_dir}")


def main(argv: list[str] | None = None) -> int:
    """Detect on processed frames or a video; write detections.json for Epic 3."""
    parser = argparse.ArgumentParser(
        description="Detect vehicles on ingested frames or a video (YOLO OBB)"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config for detection defaults (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Processed video id (reads data/processed/<id>/frames)",
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="Explicit frames directory (overrides --video-id layout)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Video file to decode when frames have not been ingested",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Traffilytics weights (default: detection.weights / models/your_obb.pt)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="detections.json path (default: data/processed/<video_id>/detections.json)",
    )
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Override processed output root (default: data.processed from config)",
    )
    parser.add_argument(
        "--overlays",
        action="store_true",
        help="Write qualitative overlays under data/processed/<video_id>/det_overlays/",
    )
    parser.add_argument(
        "--overlay-dir",
        default=None,
        help="Override overlay directory (implies --overlays)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (default from config)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold (default from config)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (default from config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (auto, cpu, 0). Default from config.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Video-only: keep every Nth frame (default: video.frame_stride)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Video-only: optional cap on decoded frames",
    )
    parser.add_argument(
        "--allow-pretrained",
        action="store_true",
        help="CPU wiring only: fall back to yolo11n-obb.pt (not FR-DET-001)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths without loading weights or running inference",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    defaults = _detection_defaults(cfg)
    video_cfg = cfg.get("video") or {}
    weights_arg = args.weights or defaults["weights"]
    weights_path = _resolve_weights_path(str(weights_arg), _REPO_ROOT)
    allow_pretrained = args.allow_pretrained or defaults["allow_pretrained"]
    device = args.device if args.device is not None else defaults["device"]

    processed_root = (
        Path(args.processed_root).resolve()
        if args.processed_root
        else resolve_data_path(cfg, "processed")
    )

    if args.frames and args.video:
        print("ERROR: pass only one of --frames or --video", file=sys.stderr)
        return 1

    video_id = args.video_id
    frames_dir: Path | None = None
    video_path: Path | None = None
    n_frames: int | None = None

    try:
        if args.frames:
            frames_dir = _resolve_user_path(args.frames, _REPO_ROOT)
            video_id = video_id or frames_dir.parent.name
            if not args.dry_run or frames_dir.exists():
                listed = list_frame_images(frames_dir)
                n_frames = len(listed)
        elif args.video:
            video_path = _resolve_user_path(args.video, _REPO_ROOT)
            video_id = video_id or video_path.stem
            if not video_path.is_file() and not args.dry_run:
                raise DetectorError(f"Video file not found: {video_path}")
        elif args.video_id:
            frames_dir = processed_root / args.video_id / "frames"
            if not args.dry_run or frames_dir.exists():
                listed = list_frame_images(frames_dir)
                n_frames = len(listed)
        elif not args.dry_run:
            raise DetectorError(
                "Specify --video-id, --frames, or --video. "
                "Example: python scripts/detect_frames.py --video-id <id>"
            )
    except DetectorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    dest = (
        _resolve_user_path(args.out, _REPO_ROOT)
        if args.out
        else (
            (processed_root / video_id / DEFAULT_DETECTIONS_NAME)
            if video_id
            else None
        )
    )
    overlay_dir: Path | None = None
    if args.overlay_dir:
        overlay_dir = _resolve_user_path(args.overlay_dir, _REPO_ROOT)
    elif args.overlays and video_id:
        overlay_dir = processed_root / video_id / DEFAULT_OVERLAY_DIRNAME

    if args.dry_run:
        source_kind = "frames" if frames_dir is not None else (
            "video" if video_path is not None else None
        )
        source_path = frames_dir or video_path
        if source_path is not None and not source_path.exists():
            print(f"ERROR: Source not found: {source_path}", file=sys.stderr)
            return 1
        _print_dry_run(
            weights=weights_path,
            weights_missing=not weights_path.is_file(),
            allow_pretrained=allow_pretrained,
            source_kind=source_kind,
            source_path=source_path,
            n_frames=n_frames,
            dest=dest,
            overlay_dir=overlay_dir,
            video_id=video_id,
            device=device,
        )
        return 0

    if dest is None:
        print("ERROR: Could not resolve detections.json output path", file=sys.stderr)
        return 1

    try:
        detector = VehicleDetector(
            weights_path,
            conf_threshold=(
                args.conf if args.conf is not None else defaults["conf_threshold"]
            ),
            iou_threshold=(
                args.iou if args.iou is not None else defaults["iou_threshold"]
            ),
            imgsz=args.imgsz if args.imgsz is not None else defaults["imgsz"],
            device=device,
            allow_pretrained=allow_pretrained,
        )
        if frames_dir is not None:
            result = detect_and_write(
                detector,
                list_frame_images(frames_dir),
                dest,
                video_id=video_id,
                overlay_dir=overlay_dir,
                weights=str(weights_path),
            )
        elif video_path is not None:
            stride = (
                args.stride
                if args.stride is not None
                else int(video_cfg.get("frame_stride") or 1)
            )
            max_frames = args.max_frames
            if max_frames is None and video_cfg.get("max_frames") is not None:
                max_frames = int(video_cfg["max_frames"])
            result = detect_video(
                detector,
                video_path,
                dest,
                video_id=video_id,
                overlay_dir=overlay_dir,
                weights=str(weights_path),
                stride=stride,
                max_frames=max_frames,
            )
        else:
            raise DetectorError("Specify --video-id, --frames, or --video")
    except DetectorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
