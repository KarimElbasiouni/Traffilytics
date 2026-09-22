"""Build Traffilytics trajectories from tracked detections (FR-TRK-003)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from computer_vision.tracking.types import TrackedDetection
from computer_vision.trajectories.lanes import LaneAssigner
from computer_vision.trajectories.types import Trajectory, TrajectoryPoint

DEFAULT_TRAJECTORIES_NAME = "trajectories.json"


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
