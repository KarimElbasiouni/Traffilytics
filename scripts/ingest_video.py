#!/usr/bin/env python3
"""CLI: ingest a traffic video through Traffilytics VideoProcessor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without install when invoked from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from computer_vision.preprocessing.config import load_config, resolve_data_path
from computer_vision.preprocessing.video_processor import VideoProcessor, VideoProcessorError


def build_parser() -> argparse.ArgumentParser:
    """Define CLI flags for video path, config, identity overrides, and frame limits."""
    parser = argparse.ArgumentParser(
        description="Ingest a traffic video: extract frames and write metadata.json"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video (e.g. data/raw/site_03_clip_01.mp4)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument("--video-id", default=None, help="Override video_id (default: filename stem)")
    parser.add_argument("--site", default=None, help="Override site id")
    parser.add_argument(
        "--source",
        default=None,
        help="Override source label (default from config ingest.default_source)",
    )
    parser.add_argument(
        "--stabilized",
        choices=("true", "false", "auto"),
        default="auto",
        help="Whether input is pre-stabilized (default: auto from filename)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frame stride (default: video.frame_stride from config)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on extracted frames",
    )
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Override processed output root (default: data.processed from config)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ingest: load config, process the video, print metadata, return exit code."""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    video_cfg = cfg.get("video") or {}
    ingest_cfg = cfg.get("ingest") or {}

    stride = args.stride if args.stride is not None else int(video_cfg.get("frame_stride") or 1)
    max_frames = args.max_frames
    if max_frames is None and video_cfg.get("max_frames") is not None:
        max_frames = int(video_cfg["max_frames"])

    source = args.source or ingest_cfg.get("default_source") or "drift"
    infer_stab = bool(ingest_cfg.get("infer_stabilized_from_name", True))

    stabilized: bool | None
    if args.stabilized == "auto":
        stabilized = None
    else:
        stabilized = args.stabilized == "true"

    processed_root = (
        Path(args.processed_root).resolve()
        if args.processed_root
        else resolve_data_path(cfg, "processed")
    )

    image_format = str(video_cfg.get("frame_image_format") or "jpg")
    jpeg_quality = int(video_cfg.get("jpeg_quality") or 90)

    try:
        with VideoProcessor(
            args.video,
            video_id=args.video_id,
            site=args.site,
            source=source,
            stabilized=stabilized,
            infer_stabilized_from_name=infer_stab,
        ) as processor:
            result = processor.process(
                processed_root,
                stride=stride,
                max_frames=max_frames,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
    except VideoProcessorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result["metadata"], indent=2))
    print(
        f"\nWrote {result['frames_written']} frames → {result['frames_dir']}\n"
        f"Metadata → {result['metadata_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
