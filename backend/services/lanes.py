"""Load and store per-video lane/zone polygons (FR-TRK-006, FR-UI-004)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from computer_vision.trajectories.lanes import LaneAssigner, LaneConfigError


def parse_lane_payload(payload: Mapping[str, Any], *, video_id: str) -> LaneAssigner:
    """Validate an API/file body and stamp ``video_id``."""
    data = dict(payload)
    data["video_id"] = video_id
    return LaneAssigner.from_mapping(data)


def write_lane_config(
    assigner: LaneAssigner,
    *,
    video_id: str,
    lanes_dir: Path,
) -> Path:
    """Write ``configs/lanes/<video_id>.json`` so CLI + worker share one file."""
    lanes_dir.mkdir(parents=True, exist_ok=True)
    dest = lanes_dir / f"{video_id}.json"
    payload = assigner.to_dict()
    payload["video_id"] = video_id
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assigner.source = dest
    return dest


def load_lane_config(video_id: str, *, lanes_dir: Path) -> LaneAssigner | None:
    return LaneAssigner.discover(video_id, lanes_dir=lanes_dir)


def lanes_from_video_json(lane_config: Mapping[str, Any] | None, *, video_id: str) -> LaneAssigner | None:
    if not lane_config:
        return None
    try:
        return parse_lane_payload(lane_config, video_id=video_id)
    except LaneConfigError:
        return None
