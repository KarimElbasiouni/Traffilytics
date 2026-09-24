"""Template-based insight text (FR-INS-001, FR-INS-002). LLM later (NFR-MAINT-003)."""

from __future__ import annotations

from typing import Sequence

from analytics.types import (
    BottleneckResult,
    FlowResult,
    ImbalanceResult,
    Insight,
    TrafficEvent,
)


def generate_insights(
    *,
    video_id: str,
    n_trajectories: int,
    units_note: str,
    speed_unit: str,
    flow: FlowResult,
    bottleneck: BottleneckResult,
    imbalance: ImbalanceResult,
    events: Sequence[TrafficEvent],
) -> list[Insight]:
    """Turn structured analytics into short readable sentences."""
    insights: list[Insight] = []
    if n_trajectories == 0:
        insights.append(Insight(id="empty", text=f"Scene {video_id}: no trajectories to analyze."))
        return insights

    speed = f"{flow.mean_speed:.1f} {speed_unit}" if flow.mean_speed is not None else "n/a"
    insights.append(
        Insight(
            id="overview",
            text=(
                f"Scene {video_id}: {n_trajectories} tracks, "
                f"{flow.vehicles_per_minute:.1f} vehicles/min, mean speed {speed}. "
                f"Units: {units_note}."
            ),
        )
    )

    if bottleneck.configured and bottleneck.primary:
        primary = bottleneck.primary
        insights.append(
            Insight(
                id="bottleneck",
                text=(
                    f"Primary bottleneck: {primary.get('zone')} "
                    f"(cause: {primary.get('cause')})."
                ),
            )
        )
    else:
        insights.append(
            Insight(
                id="bottleneck",
                text="No analysis zones configured; bottleneck location was not inferred.",
            )
        )

    if imbalance.configured and imbalance.lanes:
        ranked = sorted(imbalance.lanes, key=lambda r: r.share, reverse=True)
        top = ranked[0]
        if len(ranked) >= 2:
            second = ranked[1]
            insights.append(
                Insight(
                    id="imbalance",
                    text=(
                        f"Lane utilization imbalance: {top.lane} carried "
                        f"{top.share:.0%} of assigned tracks vs {second.lane} "
                        f"{second.share:.0%} (index {imbalance.imbalance_index:.2f})."
                    ),
                )
            )
        else:
            insights.append(
                Insight(
                    id="imbalance",
                    text=f"Only one lane observed ({top.lane}); no pair to compare.",
                )
            )
    else:
        insights.append(
            Insight(
                id="imbalance",
                text="No lane polygons configured; lane utilization was not computed.",
            )
        )

    stopped = sum(1 for e in events if e.type == "stopped_vehicle")
    sudden = sum(1 for e in events if e.type == "sudden_congestion")
    spill = sum(1 for e in events if e.type == "queue_spillback")
    insights.append(
        Insight(
            id="events",
            text=(
                f"Events: {stopped} stopped vehicle(s), "
                f"{sudden} sudden congestion, {spill} queue spillback."
            ),
        )
    )
    return insights

