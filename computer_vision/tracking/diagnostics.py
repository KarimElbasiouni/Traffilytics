"""Tracking quality without trajectory ground truth (FR-TRK-005).

UAV-OBB has no GT trajectories, so these figures are **diagnostics**, not MOTA.
Suspected ID switches are spatial/temporal handoffs: a track ends and another of
the same class starts nearby shortly after. Fragmentation is a gap in a track's
own frame sequence larger than the inferred frame step.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from computer_vision.trajectories.types import Trajectory

DEFAULT_DIAGNOSTICS_NAME = "tracking_diagnostics.json"
_NO_GT_NOTE = (
    "UAV-OBB has no trajectory ground truth. These figures are platform "
    "diagnostics, not MOTA/IDF1. suspected_id_switches counts same-class "
    "handoffs (a track ends and another starts nearby); they are a proxy."
)


def infer_frame_step(trajectories: Sequence[Trajectory], *, default: int = 1) -> int:
    """GCD of consecutive unique frame indices across all points; ``default`` if unknown."""
    frames = sorted({p.frame for t in trajectories for p in t.points})
    diffs = [b - a for a, b in zip(frames, frames[1:]) if b > a]
    if not diffs:
        return default
    step = diffs[0]
    for d in diffs[1:]:
        step = math.gcd(step, d)
    return max(step, default)


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class TrackingDiagnostics:
    """Stability report over generated trajectories (no GT required)."""

    id_switch_max_gap: int = 5
    id_switch_max_dist: float = 80.0

    def fragmentation(self, trajectories: Sequence[Trajectory], *, frame_step: int) -> dict[str, Any]:
        """Count intra-track frame gaps larger than ``frame_step``."""
        n_gaps = 0
        fragmented = 0
        for traj in trajectories:
            frames = [p.frame for p in traj.points]
            had_gap = False
            for a, b in zip(frames, frames[1:]):
                if b - a > frame_step:
                    n_gaps += 1
                    had_gap = True
            if had_gap:
                fragmented += 1
        return {
            "frame_step": frame_step,
            "n_gaps": n_gaps,
            "fragmented_tracks": fragmented,
        }

    def suspected_handoffs(
        self,
        trajectories: Sequence[Trajectory],
        *,
        frame_step: int,
    ) -> list[dict[str, Any]]:
        """Greedy same-class handoffs: ended track → new track nearby within a short gap."""
        max_gap = self.id_switch_max_gap * frame_step
        ends: list[tuple[Trajectory, Any]] = []
        starts: list[tuple[Trajectory, Any]] = []
        for traj in trajectories:
            if not traj.points:
                continue
            ends.append((traj, traj.points[-1]))
            starts.append((traj, traj.points[0]))

        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        for i, (ended, last) in enumerate(ends):
            for j, (begun, first) in enumerate(starts):
                if ended.track_id == begun.track_id:
                    continue
                if ended.class_id != begun.class_id:
                    continue
                dt = first.frame - last.frame
                if dt <= 0 or dt > max_gap:
                    continue
                dist = math.hypot(first.center_x - last.center_x, first.center_y - last.center_y)
                if dist > self.id_switch_max_dist:
                    continue
                event = {
                    "from_track_id": ended.track_id,
                    "to_track_id": begun.track_id,
                    "class_id": ended.class_id,
                    "end_frame": last.frame,
                    "start_frame": first.frame,
                    "distance_px": round(dist, 2),
                }
                candidates.append((dist, i, j, event))

        candidates.sort(key=lambda row: row[0])
        used_end: set[int] = set()
        used_start: set[int] = set()
        accepted: list[dict[str, Any]] = []
        for _, i, j, event in candidates:
            if i in used_end or j in used_start:
                continue
            used_end.add(i)
            used_start.add(j)
            accepted.append(event)
        return accepted

    def report(self, trajectories: Sequence[Trajectory]) -> dict[str, Any]:
        """JSON-serializable stability summary (FR-TRK-005)."""
        lengths = [len(t.points) for t in trajectories]
        n_tracks = len(trajectories)
        n_points = sum(lengths)
        frame_step = infer_frame_step(trajectories)
        frag = self.fragmentation(trajectories, frame_step=frame_step)
        handoffs = self.suspected_handoffs(trajectories, frame_step=frame_step)
        payload: dict[str, Any] = {
            "n_tracks": n_tracks,
            "n_points": n_points,
            "mean_track_length": (n_points / n_tracks) if n_tracks else 0.0,
            "median_track_length": _median(lengths),
            "singleton_tracks": sum(1 for n in lengths if n == 1),
            "max_track_length": max(lengths) if lengths else 0,
            "suspected_id_switches": len(handoffs),
            "handoffs": handoffs,
            "note": _NO_GT_NOTE,
        }
        payload.update(frag)
        return payload

    def write_json(
        self,
        trajectories: Sequence[Trajectory],
        dest: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write ``tracking_diagnostics.json`` next to trajectories by default."""
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.report(trajectories)
        if extra:
            payload = {**extra, **payload}
        dest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return dest_path
