#!/usr/bin/env python3
"""Pack one finished clip for the public site.

Writes a tarball whose paths are relative to the repo root:

    data/traffilytics.db
    data/raw/<video_id>.mp4
    data/processed/<video_id>/

The database contains only that clip, so a fresh host does not list videos
whose files are not in the bundle.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import tarfile
from pathlib import Path

_TABLES = (
    "videos",
    "vehicles",
    "analytics",
    "events",
    "evaluation_runs",
    "jobs",
    "reports",
    "trajectories",
)


def export_bundle(repo: Path, video_id: str, dest: Path) -> None:
    src_db = repo / "data" / "traffilytics.db"
    raw = repo / "data" / "raw" / f"{video_id}.mp4"
    processed = repo / "data" / "processed" / video_id
    if not src_db.is_file():
        raise SystemExit(f"Missing database: {src_db}")
    if not raw.is_file():
        raise SystemExit(f"Missing raw clip: {raw}")
    if not processed.is_dir():
        raise SystemExit(f"Missing processed dir: {processed}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{dest.stem}-staging"
    if staging.exists():
        raise SystemExit(f"Staging directory already exists: {staging}")
    data = staging / "data"
    (data / "raw").mkdir(parents=True)
    (data / "processed").mkdir()

    db_out = data / "traffilytics.db"
    src = sqlite3.connect(src_db)
    schema = [
        line
        for line in src.iterdump()
        if not line.startswith("INSERT INTO")
    ]
    dst = sqlite3.connect(db_out)
    dst.executescript("\n".join(schema))
    dst.execute("ATTACH DATABASE ? AS src", (str(src_db),))
    for table in _TABLES:
        dst.execute(
            f"INSERT INTO {table} SELECT * FROM src.{table} WHERE video_id = ?",
            (video_id,),
        )
    dst.commit()
    n = dst.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    dst.close()
    src.close()
    if n != 1:
        raise SystemExit(f"Expected 1 video row, found {n}")

    (data / "raw" / raw.name).write_bytes(raw.read_bytes())
    # copytree via tar of the processed directory
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(data / "traffilytics.db", arcname="data/traffilytics.db")
        tar.add(data / "raw" / raw.name, arcname=f"data/raw/{raw.name}")
        tar.add(processed, arcname=f"data/processed/{video_id}")

    shutil.rmtree(staging)
    print(f"Wrote {dest}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export one finished clip as a public demo bundle")
    parser.add_argument("--video-id", default="test_video2")
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo / "data" / "demo-test_video2.tar.gz",
    )
    args = parser.parse_args()
    export_bundle(repo, args.video_id, args.dest)


if __name__ == "__main__":
    main()
