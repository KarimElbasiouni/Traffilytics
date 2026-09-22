"""Multi-object tracking integration (ByteTrack primary) and track_id assignment."""

from computer_vision.tracking.bytetrack import ByteTrackBackend, TrackerError
from computer_vision.tracking.diagnostics import TrackingDiagnostics
from computer_vision.tracking.overlay import (
    draw_tracked_overlay,
    write_overlay_stills,
    write_overlay_video,
)
from computer_vision.tracking.tracker import VehicleTracker
from computer_vision.tracking.types import TrackedDetection

__all__ = [
    "ByteTrackBackend",
    "TrackedDetection",
    "TrackerError",
    "TrackingDiagnostics",
    "VehicleTracker",
    "draw_tracked_overlay",
    "write_overlay_stills",
    "write_overlay_video",
]
