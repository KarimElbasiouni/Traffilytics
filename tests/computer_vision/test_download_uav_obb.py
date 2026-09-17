"""CPU/offline tests for UAV-OBB acquisition helpers and scripts/download_uav_obb.py.

The real download is never exercised here (NFR-TEST-005): metadata parsing runs
against a captured Mendeley payload, and the CLI is driven through ``--src`` with
a synthetic dataset tree, so nothing touches the network.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from adapters.uav_obb.acquire import (
    MENDELEY_DATASET_ID,
    find_bundled_videos,
    find_dataset_root,
    has_splits,
    install_splits,
    install_videos,
    mendeley_files_url,
    parse_file_listing,
    sha256_of,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_TIMEOUT_S = 60

# Shape captured from the live public listing for dataset 6snrjwcpkh v3.
_LISTING = [
    {
        "filename": "UAV-OBB.zip",
        "id": "dd0dd3f2-8138-4236-87f0-92c15b2589ec",
        "size": 530039097,
        "content_details": {
            "sha256_hash": "5bebd2979b5a20cd6df0c14241341c9bf001df07f3a6fce37ea02735e6e4bcff",
            "content_type": "application/x-zip-compressed",
            "size": 530039097,
            "download_url": "https://data.mendeley.com/public-files/datasets/6snrjwcpkh/f/file_downloaded",
        },
    }
]


def _make_dataset(root: Path, *, videos: bool = True) -> Path:
    """Build a miniature UAV-OBB tree nested under a ``UAV-OBB/`` folder."""
    base = root / "UAV-OBB"
    for split, stem in (("train", "a"), ("valid", "b"), ("test", "c")):
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
        (base / split / "images" / f"{stem}.jpg").write_bytes(b"jpg-bytes")
        (base / split / "labels" / f"{stem}.txt").write_text(
            "2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
        )
    (base / "data.yaml").write_text("names:\n  2: car\n", encoding="utf-8")
    if videos:
        (base / "test_videos_mp4").mkdir(parents=True, exist_ok=True)
        (base / "test_videos_mp4" / "clip1.mp4").write_bytes(b"mp4-one")
        (base / "test_videos_mp4" / "clip2.mp4").write_bytes(b"mp4-two")
        (base / "test_videos_mp4" / "notes.txt").write_text("ignore me", encoding="utf-8")
    return base


def _write_config(path: Path, data_root: Path) -> Path:
    """Write a config whose data.* paths are absolute and inside the tmp dir."""
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {data_root}",
                f"  raw: {data_root / 'raw'}",
                f"  annotations: {data_root / 'annotations'}",
                f"  processed: {data_root / 'processed'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run scripts/download_uav_obb.py from the repo root with the test interpreter."""
    return subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "download_uav_obb.py"), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_S,
        check=False,
    )


def test_mendeley_files_url_targets_the_public_listing() -> None:
    url = mendeley_files_url()
    assert MENDELEY_DATASET_ID in url
    assert url.startswith("https://")
    assert "version=3" in url


def test_parse_file_listing_reads_url_size_and_checksum() -> None:
    remote = parse_file_listing(_LISTING)
    assert remote.filename == "UAV-OBB.zip"
    assert remote.size == 530039097
    assert remote.sha256.startswith("5bebd297")
    assert remote.download_url.endswith("file_downloaded")


@pytest.mark.parametrize(
    "payload",
    [
        {"filename": "UAV-OBB.zip"},
        [],
        [{"filename": "readme.txt", "content_details": {"download_url": "https://x"}}],
        [{"filename": "UAV-OBB.zip", "content_details": {}}],
    ],
)
def test_parse_file_listing_rejects_unusable_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        parse_file_listing(payload)


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"traffilytics" * 1000)
    assert sha256_of(blob) == hashlib.sha256(b"traffilytics" * 1000).hexdigest()


def test_has_splits_accepts_valid_or_val(tmp_path: Path) -> None:
    (tmp_path / "a" / "train").mkdir(parents=True)
    (tmp_path / "a" / "valid").mkdir()
    (tmp_path / "b" / "train").mkdir(parents=True)
    (tmp_path / "b" / "val").mkdir()
    (tmp_path / "c" / "train").mkdir(parents=True)

    assert has_splits(tmp_path / "a")
    assert has_splits(tmp_path / "b")
    assert not has_splits(tmp_path / "c")


def test_find_dataset_root_handles_nested_and_flat_trees(tmp_path: Path) -> None:
    nested = _make_dataset(tmp_path / "nested")
    assert find_dataset_root(tmp_path / "nested") == nested.resolve()
    assert find_dataset_root(nested) == nested.resolve()


def test_find_dataset_root_errors_without_splits(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path / "missing")


def test_install_splits_copies_splits_and_data_yaml(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "src")
    dest = tmp_path / "annotations"

    copied = install_splits(source, dest)

    assert copied == ["train", "valid", "test", "data.yaml"]
    assert (dest / "train" / "images" / "a.jpg").read_bytes() == b"jpg-bytes"
    assert (dest / "valid" / "labels" / "b.txt").is_file()
    assert (dest / "data.yaml").read_text(encoding="utf-8").strip().startswith("names:")


def test_install_splits_is_idempotent(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "src")
    dest = tmp_path / "annotations"

    assert install_splits(source, dest) == install_splits(source, dest)
    assert (dest / "test" / "images" / "c.jpg").is_file()


def test_install_splits_errors_when_no_splits_present(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(FileNotFoundError):
        install_splits(bare, tmp_path / "annotations")


def test_find_bundled_videos_skips_non_video_files(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "src")
    (source / "extra.mov").write_bytes(b"mov")

    names = [p.name for p in find_bundled_videos(source)]

    assert names == ["extra.mov", "clip1.mp4", "clip2.mp4"]


def test_install_videos_copies_clips_into_raw(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "src")
    raw = tmp_path / "raw"

    written = install_videos(source, raw)

    assert sorted(p.name for p in written) == ["clip1.mp4", "clip2.mp4"]
    assert (raw / "clip1.mp4").read_bytes() == b"mp4-one"
    assert not (raw / "notes.txt").exists()


def test_install_videos_returns_empty_without_clips(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "src", videos=False)
    assert install_videos(source, tmp_path / "raw") == []


def test_cli_installs_from_local_directory(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "src")
    data_root = tmp_path / "data"
    config = _write_config(tmp_path / "cfg.yaml", data_root)

    result = _run_cli(["--config", str(config), "--src", str(tmp_path / "src")])

    assert result.returncode == 0, result.stderr
    assert (data_root / "annotations" / "train" / "images" / "a.jpg").is_file()
    assert (data_root / "annotations" / "data.yaml").is_file()
    assert (data_root / "raw" / "clip1.mp4").is_file()
    assert "prepare_obb_dataset.py" in result.stdout


def test_cli_installs_from_local_zip_and_stages_cleanly(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "src")
    archive = Path(
        shutil.make_archive(
            str(tmp_path / "UAV-OBB"), "zip", root_dir=tmp_path / "src", base_dir="UAV-OBB"
        )
    )
    data_root = tmp_path / "data"
    config = _write_config(tmp_path / "cfg.yaml", data_root)

    result = _run_cli(["--config", str(config), "--src", str(archive), "--skip-videos"])

    assert result.returncode == 0, result.stderr
    assert (data_root / "annotations" / "valid" / "images" / "b.jpg").is_file()
    assert not (data_root / "raw" / "clip1.mp4").exists()
    assert not list(data_root.glob(".uav_obb_stage_*")), "staging dir left behind"


def test_cli_dry_run_from_local_source_writes_nothing(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "src")
    data_root = tmp_path / "data"
    config = _write_config(tmp_path / "cfg.yaml", data_root)

    result = _run_cli(
        ["--config", str(config), "--src", str(tmp_path / "src"), "--dry-run"]
    )

    assert result.returncode == 0, result.stderr
    assert "Would install" in result.stdout
    assert not (data_root / "annotations").exists()


def test_cli_reports_missing_source(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "cfg.yaml", tmp_path / "data")

    result = _run_cli(["--config", str(config), "--src", str(tmp_path / "nope.zip")])

    assert result.returncode == 1
    assert "--src not found" in result.stderr
