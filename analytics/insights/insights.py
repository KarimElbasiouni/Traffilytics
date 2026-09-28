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
    _ = units_note

    speed = f"{flow.mean_speed:.1f} {speed_unit}" if flow.mean_speed is not None else "n/a"
    insights.append(
        Insight(
            id="overview",
            text=(
                f"{n_trajectories} tracks, "
                f"{flow.vehicles_per_minute:.1f} vehicles/min, mean speed {speed}."
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
    elif bottleneck.configured:
        insights.append(
            Insight(
                id="bottleneck",
                text="Zones are configured; no bottleneck stood out.",
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

    stopped = sum(1 for e in events if e.type == "stopped_vehicle")
    sudden = sum(1 for e in events if e.type == "sudden_congestion")
    spill = sum(1 for e in events if e.type == "queue_spillback")
    total_events = stopped + sudden + spill
    if total_events:
        insights.append(
            Insight(
                id="events",
                text=(
                    f"Events: {stopped} stopped vehicle(s), "
                    f"{sudden} sudden congestion, {spill} queue spillback."
                ),
            )
        )
    else:
        insights.append(
            Insight(
                id="events",
                text="No stopped vehicles, sudden congestion, or queue spillback.",
            )
        )
    return insights

