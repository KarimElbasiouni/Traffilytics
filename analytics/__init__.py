"""Analytics engine package (Epic 4): flow, bottleneck, imbalance, events, insights."""

from analytics.engine import AnalyticsEngine, merge_analytics_cfg
from analytics.types import (
    DEFAULT_ANALYTICS_NAME,
    AnalyticsError,
    AnalyticsReport,
    SceneContext,
    UnitSystem,
)

__all__ = [
    "DEFAULT_ANALYTICS_NAME",
    "AnalyticsEngine",
    "AnalyticsError",
    "AnalyticsReport",
    "SceneContext",
    "UnitSystem",
    "merge_analytics_cfg",
]
