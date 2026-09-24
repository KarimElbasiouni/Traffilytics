"""Rule-based traffic event detection (stopped vehicle, sudden congestion, spillback)."""

from analytics.events.events import detect_events

__all__ = ["detect_events"]
