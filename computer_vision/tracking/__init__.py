"""Multi-object tracking integration (ByteTrack primary) and track_id assignment."""

from computer_vision.tracking.bytetrack import ByteTrackBackend, TrackerError
from computer_vision.tracking.tracker import VehicleTracker
from computer_vision.tracking.types import TrackedDetection

__all__ = [
    "ByteTrackBackend",
    "TrackedDetection",
    "TrackerError",
    "VehicleTracker",
]
