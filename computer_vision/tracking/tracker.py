"""VehicleTracker: associate OBB detections across frames (FR-TRK-001, FR-TRK-002)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from computer_vision.detection.types import Detection
from computer_vision.tracking.bytetrack import ByteTrackBackend
from computer_vision.tracking.types import TrackedDetection


class TrackerBackend(Protocol):
    """One frame of detections in, those that received a ``track_id`` out."""

    def update(self, detections: list[Detection]) -> list[TrackedDetection]: ...


class VehicleTracker:
    """Run a tracker backend over a time-ordered detection stream.

    Default backend is Ultralytics ByteTrack. Inject ``backend`` in tests to
    avoid importing torch. Empty frames still call ``update`` so lost tracks age.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        *,
        backend: TrackerBackend | None = None,
    ) -> None:
        self.cfg = dict(cfg or {})
        self._backend: TrackerBackend = backend if backend is not None else ByteTrackBackend(self.cfg)

    def initialize_tracker(self) -> None:
        """Reset backend state (ByteTrack id counter and active tracks)."""
        reset = getattr(self._backend, "reset", None)
        if callable(reset):
            reset()

    def update_tracks(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Associate one frame. All ``detections`` must share the same ``frame``."""
        return self._backend.update(list(detections))

    def track(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Walk frames in order (including empty ones between min and max)."""
        if not detections:
            return []
        by_frame: dict[int, list[Detection]] = {}
        for det in detections:
            by_frame.setdefault(det.frame, []).append(det)
        out: list[TrackedDetection] = []
        for frame in range(min(by_frame), max(by_frame) + 1):
            out.extend(self.update_tracks(by_frame.get(frame, [])))
        return out
