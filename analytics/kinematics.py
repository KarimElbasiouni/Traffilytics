"""Point kinematics from generated trajectories (speed, heading, zone)."""

from __future__ import annotations

import math
from typing import Sequence

from analytics.types import Sample, SceneContext
from computer_vision.trajectories.types import Trajectory


def build_samples(
    trajectories: Sequence[Trajectory],
    context: SceneContext,
) -> list[Sample]:
    """Attach speed/heading (and optional zone) to each trajectory point.

    Speed on the first point of a track is copied from the first segment when
    one exists; singleton tracks stay ``speed=None`` so they are not treated
    as stopped vehicles.
    """
    assigner = context.assigner
    fps = float(context.fps)
    if fps <= 0:
        from analytics.types import AnalyticsError

        raise AnalyticsError(f"fps must be > 0, got {fps}")

    samples: list[Sample] = []
    for traj in trajectories:
        points = sorted(traj.points, key=lambda p: p.frame)
        speeds: list[float | None] = [None] * len(points)
        headings: list[float | None] = [None] * len(points)
        for i in range(1, len(points)):
            prev, cur = points[i - 1], points[i]
            dt = (cur.frame - prev.frame) / fps
            if dt <= 0:
                continue
            dist = math.hypot(cur.center_x - prev.center_x, cur.center_y - prev.center_y)
            speeds[i] = context.units.speed_from_pixels(dist, dt)
            headings[i] = math.atan2(cur.center_y - prev.center_y, cur.center_x - prev.center_x)
        if len(points) >= 2 and speeds[1] is not None:
            speeds[0] = speeds[1]
            headings[0] = headings[1]

        for point, speed, heading in zip(points, speeds, headings):
            lane = point.lane if point.lane is not None else traj.lane
            zone = None
            if assigner is not None:
                zone = assigner.assign_zone(point.center_x, point.center_y)
                if lane is None:
                    lane = assigner.assign_lane(point.center_x, point.center_y)
            samples.append(
                Sample(
                    track_id=traj.track_id,
                    frame=point.frame,
                    t=point.frame / fps,
                    x=point.center_x,
                    y=point.center_y,
                    speed=speed,
                    heading=heading,
                    lane=lane,
                    zone=zone,
                    class_id=traj.class_id,
                    video_id=traj.video_id or context.video_id,
                )
            )
    samples.sort(key=lambda s: (s.frame, s.track_id))
    return samples


def samples_in_window(
    samples: Sequence[Sample],
    t0: float,
    t1: float,
    *,
    t_max: float,
) -> list[Sample]:
    """Half-open ``[t0, t1)`` except the last window, which is closed on the right."""
    if t1 >= t_max:
        return [s for s in samples if t0 <= s.t <= t1]
    return [s for s in samples if t0 <= s.t < t1]


def window_bounds(t_min: float, t_max: float, window_seconds: float) -> list[tuple[float, float]]:
    width = max(float(window_seconds), 1e-6)
    if t_max <= t_min:
        return [(t_min, t_min + width)]
    bounds: list[tuple[float, float]] = []
    t = t_min
    while t < t_max:
        end = min(t + width, t_max)
        if end > t:
            bounds.append((t, end))
        t += width
        if len(bounds) > 10_000:
            break
    return bounds or [(t_min, t_max)]
