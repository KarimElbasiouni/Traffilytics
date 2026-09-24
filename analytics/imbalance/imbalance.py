"""Dedicated lane utilization / imbalance (FR-IMB-001, FR-IMB-002)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from analytics.types import ImbalanceResult, LaneShare, Sample, SceneContext, mean


def analyze_imbalance(
    samples: Sequence[Sample],
    context: SceneContext,
    cfg: Mapping[str, Any],
) -> ImbalanceResult:
    """Share of tracks per configured (or stamped) lane, plus mean heading."""
    del cfg  # thresholds live on the flow/event side; imbalance is a census
    assigner = context.assigner
    configured_ids = [lane.id for lane in assigner.lanes] if assigner is not None else []
    stamped = sorted({s.lane for s in samples if s.lane})
    lane_ids = configured_ids or stamped
    if not lane_ids:
        return ImbalanceResult(
            configured=False,
            note="No lane polygons and no stamped trajectory.lane values; imbalance skipped.",
        )

    tracks_by_lane: dict[str, set[int]] = defaultdict(set)
    headings: dict[str, list[float]] = defaultdict(list)
    speeds: dict[str, list[float]] = defaultdict(list)
    majority: dict[int, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.lane:
            majority[sample.track_id].append(sample.lane)
            if sample.heading is not None:
                headings[sample.lane].append(sample.heading)
            if sample.speed is not None:
                speeds[sample.lane].append(sample.speed)

    for track_id, votes in majority.items():
        winner = max(set(votes), key=votes.count)
        tracks_by_lane[winner].add(track_id)

    n_assigned = sum(len(v) for v in tracks_by_lane.values())
    rows: list[LaneShare] = []
    for lane_id in lane_ids:
        n = len(tracks_by_lane.get(lane_id) or ())
        share = (n / n_assigned) if n_assigned else 0.0
        rows.append(
            LaneShare(
                lane=lane_id,
                n_tracks=n,
                share=share,
                mean_heading=mean(headings.get(lane_id) or []),
                mean_speed=mean(speeds.get(lane_id) or []),
            )
        )

    shares = [row.share for row in rows]
    imbalance_index = (max(shares) - min(shares)) if shares else None
    dominant = max(rows, key=lambda r: (r.n_tracks, r.share)).lane if rows else None
    return ImbalanceResult(
        configured=True,
        lanes=rows,
        imbalance_index=imbalance_index,
        dominant_lane=dominant,
    )
