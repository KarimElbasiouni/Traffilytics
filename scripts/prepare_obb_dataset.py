#!/usr/bin/env python3
"""Prepare Ultralytics data.yaml from local DRIFT-style OBB annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adapters.drift.obb_annotations import DriftOBBDataset
from adapters.drift.drift_layout import DriftLayout


def main(argv: list[str] | None = None) -> int:
    """Scan local OBB annotations and write an Ultralytics data.yaml for training."""
    parser = argparse.ArgumentParser(description="Write YOLO data.yaml for DRIFT OBB annotations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--annotations",
        default=None,
        help="Annotations root (default: data/annotations from config)",
    )
    parser.add_argument(
        "--out",
        default="models/configs/drift_obb_data.yaml",
        help="Output Ultralytics data.yaml path",
    )
    args = parser.parse_args(argv)

    layout = DriftLayout.from_config(args.config)
    ann_root = Path(args.annotations) if args.annotations else layout.annotations
    try:
        dataset = DriftOBBDataset.from_annotations_root(ann_root)
    except FileNotFoundError as exc:
        print(
            f"ERROR: {exc}\n"
            "Layout smoke (2 frames): python scripts/download_drift_sample.py\n"
            "Full GitHub model/ splits (~2,301 frames / ~300K instances): "
            "python scripts/download_obb_dataset.py --full --src /path/to/The-DRIFT",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    written = dataset.write_ultralytics_data_yaml(out)
    train_n = dataset.count_pairs("train")
    val_n = dataset.count_pairs("val")
    print(f"Wrote {written}")
    print(f"Pairs: train={train_n} val={val_n}")
    if train_n == 0:
        print(
            "WARNING: No train image/label pairs found. "
            "Place full DRIFT OBB splits under data/annotations/ "
            "(python scripts/download_obb_dataset.py --full --src /path/to/The-DRIFT).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
