"""Load Traffilytics YAML config and resolve paths relative to the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config; default is ``configs/default.yaml`` under the repo root."""
    path = Path(config_path) if config_path else REPO_ROOT / "configs" / "default.yaml"
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_data_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a ``data.<key>`` path from config relative to the repo root."""
    data = cfg.get("data") or {}
    raw = data.get(key)
    if not raw:
        raise KeyError(f"Missing data.{key} in config")
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path
