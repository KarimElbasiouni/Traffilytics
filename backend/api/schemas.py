"""Pydantic request bodies (response payloads stay plain dicts to match the API doc)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    model_version: str = "obb_v1"
    tracker: str = "bytetrack"
    scale_m_per_px: float | None = None
    lane_config: dict[str, Any] | None = None
    reuse_artifacts: bool = True
    skip_detect: bool = False
    skip_track: bool = False


class LaneConfigBody(BaseModel):
    video_id: str | None = None
    lanes: list[dict[str, Any]] = Field(default_factory=list)
    zones: list[dict[str, Any]] = Field(default_factory=list)
