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
        """Walk ingested frames in order.

        Consecutive source indices with a hole (no detections that frame) still
        get an empty ``update`` so lost tracks can age. A regular stride (every
        2nd source frame) does **not** insert the skipped indices — those frames
        were never ingested, and empty ByteTrack updates on them drop IDs.
        """
        if not detections:
            return []
        by_frame: dict[int, list[Detection]] = {}
        for det in detections:
            by_frame.setdefault(det.frame, []).append(det)
        out: list[TrackedDetection] = []
        for frame in ingested_frame_order(by_frame):
            out.extend(self.update_tracks(by_frame.get(frame, [])))
        return out


def ingested_frame_order(by_frame: Mapping[int, Any]) -> list[int]:
    """Frame numbers to feed the tracker, without fake holes from ingest stride."""
    keys = sorted(by_frame)
    if len(keys) < 2:
        return keys
    deltas = [b - a for a, b in zip(keys, keys[1:])]
    step = min(deltas)
    if step > 1 and all(delta % step == 0 for delta in deltas):
        return keys
    return list(range(keys[0], keys[-1] + 1))
