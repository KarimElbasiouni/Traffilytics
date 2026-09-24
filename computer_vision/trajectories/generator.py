"""Build Traffilytics trajectories from tracked detections (FR-TRK-003)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from computer_vision.tracking.types import TrackedDetection
from computer_vision.trajectories.lanes import LaneAssigner
from computer_vision.trajectories.types import Trajectory, TrajectoryPoint

DEFAULT_TRAJECTORIES_NAME = "trajectories.json"


class TrajectoryError(Exception):
    """Raised when trajectories.json is missing or unreadable."""

    exit_code = 1


def summarize_tracks(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    """Counts usable without trajectory ground truth (first cut of FR-TRK-005)."""
    lengths = [len(t.points) for t in trajectories]
    n_tracks = len(trajectories)
    n_points = sum(lengths)
    return {
        "n_tracks": n_tracks,
        "n_points": n_points,
        "mean_track_length": (n_points / n_tracks) if n_tracks else 0.0,
        "singleton_tracks": sum(1 for n in lengths if n == 1),
        "max_track_length": max(lengths) if lengths else 0,
    }


@dataclass
class TrajectoryGenerator:
    """Group tracked detections by ``track_id`` into time-ordered trajectories.

    Pass a :class:`~computer_vision.trajectories.lanes.LaneAssigner` to stamp
    ``lane`` on each point (FR-TRK-006); otherwise lane stays null.
    """

    min_hits: int = 1

    def generate(
        self,
        tracked: Sequence[TrackedDetection],
        *,
        video_id: str,
        assigner: LaneAssigner | None = None,
    ) -> list[Trajectory]:
        """Return trajectories sorted by ``track_id``; drop tracks shorter than ``min_hits``."""
        buckets: dict[int, list[TrackedDetection]] = defaultdict(list)
        for item in tracked:
            buckets[item.track_id].append(item)

        trajectories: list[Trajectory] = []
        for track_id in sorted(buckets):
            items = sorted(buckets[track_id], key=lambda t: t.frame)
            if len(items) < self.min_hits:
                continue
            class_id = items[0].class_id
            points = [
                TrajectoryPoint.from_tracked(item, video_id=video_id, lane=None)
                for item in items
            ]
            trajectories.append(
                Trajectory(
                    track_id=track_id,
                    video_id=video_id,
                    class_id=class_id,
                    points=points,
                    lane=None,
                )
            )
        if assigner is not None:
            return assigner.apply(trajectories)
        return trajectories

    def write_json(
        self,
        trajectories: Sequence[Trajectory],
        dest: str | Path,
        *,
        video_id: str | None = None,
        tracker: str = "bytetrack",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write ``trajectories.json`` (grouped tracks + flat points + diagnostics)."""
        from computer_vision.tracking.diagnostics import TrackingDiagnostics

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        points = [p.to_dict() for traj in trajectories for p in traj.points]
        payload: dict[str, Any] = {
            "video_id": video_id or (trajectories[0].video_id if trajectories else None),
            "tracker": tracker,
            "lane": None,
            "diagnostics": TrackingDiagnostics().report(trajectories),
            "n_tracks": len(trajectories),
            "n_points": len(points),
            "tracks": [t.to_dict() for t in trajectories],
            "points": points,
        }
        if extra:
            payload.update(extra)
        dest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return dest_path


def load_trajectories_json(path: str | Path) -> tuple[list[Trajectory], dict[str, Any]]:
    """Read ``trajectories.json`` written by :meth:`TrajectoryGenerator.write_json`.

    Prefers the grouped ``tracks`` list; falls back to regrouping the flat
    ``points`` array. Returns ``(trajectories, metadata)`` with the payload minus
    those lists.
    """
    dest = Path(path)
    if not dest.is_file():
        raise TrajectoryError(f"Trajectories file not found: {dest}")
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrajectoryError(f"Invalid trajectories JSON: {dest} ({exc})") from exc
    if not isinstance(payload, dict):
        raise TrajectoryError(f"Trajectories JSON must be an object: {dest}")

    trajectories = _trajectories_from_payload(payload, source=dest)
    skip = {"tracks", "points"}
    meta = {k: v for k, v in payload.items() if k not in skip}
    return trajectories, meta


def _trajectories_from_payload(
    payload: Mapping[str, Any],
    *,
    source: Path,
) -> list[Trajectory]:
    raw_tracks = payload.get("tracks")
    if isinstance(raw_tracks, list) and raw_tracks:
        return [Trajectory.from_dict(row) for row in raw_tracks if isinstance(row, Mapping)]

    raw_points = payload.get("points")
    if not isinstance(raw_points, list):
        raise TrajectoryError(f"Trajectories JSON missing tracks/points: {source}")
    buckets: dict[int, list[TrajectoryPoint]] = defaultdict(list)
    for row in raw_points:
        if not isinstance(row, Mapping):
            continue
        point = TrajectoryPoint.from_dict(row)
        buckets[point.track_id].append(point)
    trajectories: list[Trajectory] = []
    default_video = str(payload.get("video_id") or "")
    for track_id in sorted(buckets):
        points = sorted(buckets[track_id], key=lambda p: p.frame)
        trajectories.append(
            Trajectory(
                track_id=track_id,
                video_id=points[0].video_id or default_video,
                class_id=points[0].class_id,
                points=points,
                lane=None,
            )
        )
    return trajectories
