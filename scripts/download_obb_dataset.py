#!/usr/bin/env python3
"""Copy full DRIFT GitHub OBB splits into data/annotations/.

The-DRIFT stores Ultralytics OBB images/labels under model/{train,valid,test}
(~2,301 frames / ~300K instances). This helper copies a *local* clone of that
tree into Traffilytics via :func:`adapters.drift.obb_annotations.sync_github_style_layout`.

The full set is gitignored and must **not** be downloaded in CI or pytest.
Use ``scripts/download_drift_sample.py`` for the two-frame layout smoke only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adapters.drift.drift_layout import DriftLayout
from adapters.drift.obb_annotations import (
    DRIFT_GITHUB_URL,
    DriftOBBDataset,
    FULL_OBB_FRAME_COUNT,
    FULL_OBB_INSTANCE_COUNT_APPROX,
    install_github_obb_splits,
    resolve_github_model_dir,
)

# Below this pair count, the dest tree still looks like a smoke sample.
_SMOKE_PAIR_WARN_THRESHOLD = 100


def build_parser() -> argparse.ArgumentParser:
    """CLI for copying a local The-DRIFT ``model/`` tree into ``data/annotations/``."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy full DRIFT GitHub OBB splits (model/{train,valid,test}) "
            "into data/annotations/"
        ),
        epilog=(
            f"Full set: ~{FULL_OBB_FRAME_COUNT:,} frames / "
            f"~{FULL_OBB_INSTANCE_COUNT_APPROX // 1000}K instances. "
            "Gitignored. Do not fetch in CI. "
            f"Clone first: {DRIFT_GITHUB_URL}"
        ),
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--src",
        default=None,
        help="Local The-DRIFT repo root or its model/ directory (required with --full)",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination annotations root (default: data/annotations from config)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Copy the full GitHub model/ splits (refused without this flag)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the copy plan without writing files",
    )
    return parser


def _print_acquisition_help() -> None:
    """Document how to obtain the full OBB tree without downloading it here."""
    print("Full DRIFT OBB annotations (not the 2-frame sample)")
    print(f"  Size:   ~{FULL_OBB_FRAME_COUNT:,} frames / ~{FULL_OBB_INSTANCE_COUNT_APPROX:,} instances")
    print(f"  Source: {DRIFT_GITHUB_URL}  (model/train, model/valid, model/test)")
    print("  Dest:   data/annotations/  (gitignored — do not commit)")
    print("  CI:     do not download this 4K set in pytest or CI")
    print()
    print("Clone The-DRIFT, then:")
    print("  python scripts/download_obb_dataset.py --full --src /path/to/The-DRIFT")
    print("  python scripts/prepare_obb_dataset.py")
    print()
    print("Layout smoke only (2 frames): python scripts/download_drift_sample.py")


def _count_pairs(ann_root: Path) -> tuple[int, int, int]:
    """Return (train, val, test) image/label pair counts; missing tree is (0, 0, 0)."""
    try:
        dataset = DriftOBBDataset.from_annotations_root(ann_root)
    except FileNotFoundError:
        return 0, 0, 0
    train_n = dataset.count_pairs("train")
    val_n = dataset.count_pairs("val")
    test_n = dataset.count_pairs("test") if dataset.test_dir is not None else 0
    return train_n, val_n, test_n


def _print_pair_counts(ann_root: Path, *, heading: str) -> tuple[int, int, int]:
    """Log split pair counts and warn when the tree still looks like a smoke sample."""
    train_n, val_n, test_n = _count_pairs(ann_root)
    total = train_n + val_n + test_n
    print(f"{heading}: train={train_n} val={val_n} test={test_n} (total={total})")
    if 0 < total < _SMOKE_PAIR_WARN_THRESHOLD:
        print(
            f"WARNING: {total} pairs looks like a layout smoke sample, "
            f"not the full ~{FULL_OBB_FRAME_COUNT:,}-frame / "
            f"~{FULL_OBB_INSTANCE_COUNT_APPROX:,}-instance set.",
            file=sys.stderr,
        )
    return train_n, val_n, test_n


def main(argv: list[str] | None = None) -> int:
    """Validate --full/--src, optionally copy GitHub-style splits, return an exit code."""
    args = build_parser().parse_args(argv)
    layout = DriftLayout.from_config(args.config)
    dest = Path(args.dest).expanduser().resolve() if args.dest else layout.annotations

    if not args.full:
        if args.src and not args.dry_run:
            print(
                "ERROR: Refusing to copy the full OBB set without --full "
                f"(~{FULL_OBB_FRAME_COUNT:,} frames / "
                f"~{FULL_OBB_INSTANCE_COUNT_APPROX:,} instances).",
                file=sys.stderr,
            )
            return 1
        _print_acquisition_help()
        print(f"Destination: {dest}")
        if args.dry_run:
            print("Dry run OK — not copying files.")
        return 0

    if not args.src:
        print(
            "ERROR: --full requires --src pointing at a local The-DRIFT clone "
            f"(repo root or model/). Clone {DRIFT_GITHUB_URL} first. "
            "This script does not download the full 4K set.",
            file=sys.stderr,
        )
        return 1

    try:
        model_dir = resolve_github_model_dir(args.src)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Source model/: {model_dir}")
    print(f"Destination:   {dest}")
    print(
        f"Full set size: ~{FULL_OBB_FRAME_COUNT:,} frames / "
        f"~{FULL_OBB_INSTANCE_COUNT_APPROX:,} instances (gitignored)"
    )

    if args.dry_run:
        print("Dry run OK — not copying files.")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = install_github_obb_splits(model_dir, dest)
    print(f"Copied GitHub model/ splits → {written}")
    _print_pair_counts(written, heading="Pairs")
    print("Next: python scripts/prepare_obb_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
