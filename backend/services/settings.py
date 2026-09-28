"""Runtime settings for the API, worker, and dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from computer_vision.preprocessing.config import REPO_ROOT, load_config, resolve_data_path

JobHandler = Callable[[str], None]


@dataclass
class Settings:
    """Paths and flags the FastAPI app and pipeline worker share."""

    repo_root: Path = REPO_ROOT
    db_url: str = "sqlite:///./data/traffilytics.db"
    data_root: Path = field(default_factory=lambda: REPO_ROOT / "data")
    raw_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "processed")
    lanes_dir: Path = field(default_factory=lambda: REPO_ROOT / "configs" / "lanes")
    models_dir: Path = field(default_factory=lambda: REPO_ROOT / "models")
    frontend_dir: Path = field(default_factory=lambda: REPO_ROOT / "frontend")
    config_path: Path = field(default_factory=lambda: REPO_ROOT / "configs" / "default.yaml")
    config: dict[str, Any] = field(default_factory=dict)
    skip_detect: bool = False
    skip_track: bool = False
    reuse_artifacts: bool = True


def load_settings(
    *,
    repo_root: Path | None = None,
    db_url: str | None = None,
    data_root: Path | None = None,
    config_path: str | Path | None = None,
    load_env: bool = True,
) -> Settings:
    """Build settings from env + ``configs/default.yaml``.

    ``db_url`` / ``data_root`` win over ``DB_URL`` / ``DATA_ROOT`` so tests can
    point at a temp tree without mutating the process environment.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    if load_env:
        load_dotenv(root / ".env")

    cfg_path = Path(config_path) if config_path else root / "configs" / "default.yaml"
    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        cfg = {}

    resolved_db = db_url or os.environ.get("DB_URL") or "sqlite:///./data/traffilytics.db"
    if resolved_db.startswith("sqlite:///./"):
        rel = resolved_db[len("sqlite:///./") :]
        resolved_db = "sqlite:///" + str((root / rel).resolve())

    if data_root is not None:
        data = Path(data_root)
        raw = data / "raw"
        processed = data / "processed"
    else:
        env_data = os.environ.get("DATA_ROOT")
        if env_data:
            data = Path(env_data)
            if not data.is_absolute():
                data = (root / data).resolve()
            raw = data / "raw"
            processed = data / "processed"
        else:
            try:
                data = resolve_data_path(cfg, "root") if (cfg.get("data") or {}).get("root") else root / "data"
            except (KeyError, FileNotFoundError):
                data = root / "data"
            try:
                raw = resolve_data_path(cfg, "raw")
            except (KeyError, FileNotFoundError):
                raw = data / "raw"
            try:
                processed = resolve_data_path(cfg, "processed")
            except (KeyError, FileNotFoundError):
                processed = data / "processed"

    lanes_cfg = cfg.get("lanes") or {}
    lanes_dir = Path(lanes_cfg.get("dir") or "configs/lanes")
    if not lanes_dir.is_absolute():
        lanes_dir = (root / lanes_dir).resolve()

    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    lanes_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        repo_root=root,
        db_url=resolved_db,
        data_root=data,
        raw_dir=raw,
        processed_dir=processed,
        lanes_dir=lanes_dir,
        models_dir=root / "models",
        frontend_dir=root / "frontend",
        config_path=cfg_path if cfg_path.is_absolute() else (root / cfg_path),
        config=cfg,
    )
