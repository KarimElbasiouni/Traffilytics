#!/usr/bin/env python3
"""Train Traffilytics YOLO OBB weights on DRIFT annotations (Epic 2).

Requires a CUDA-capable GPU for practical training. CPU may run for smoke tests
with tiny datasets but is not recommended for full DRIFT training.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_train_config(path: Path) -> dict:
    """Load the training hyperparameter YAML (model, epochs, batch, device, …)."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(value: str | Path) -> Path:
    """Turn a repo-relative path into an absolute path under the project root."""
    path = Path(value)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def main(argv: list[str] | None = None) -> int:
    """Validate paths, optionally train YOLO OBB, copy best weights to models/your_obb.pt."""
    parser = argparse.ArgumentParser(description="Train YOLO OBB on DRIFT (Traffilytics weights)")
    parser.add_argument(
        "--train-config",
        default="models/configs/train_obb.yaml",
        help="Training hyperparameter YAML",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override Ultralytics data.yaml (default from train config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (e.g. 0, cpu). Default from train config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset paths without training",
    )
    args = parser.parse_args(argv)

    train_cfg_path = resolve_path(args.train_config)
    if not train_cfg_path.is_file():
        print(f"ERROR: Missing train config: {train_cfg_path}", file=sys.stderr)
        return 1

    cfg = load_train_config(train_cfg_path)
    data_yaml = resolve_path(args.data or cfg.get("data") or "models/configs/drift_obb_data.yaml")
    model_name = cfg.get("model") or "yolov8n-obb.pt"
    device = args.device if args.device is not None else cfg.get("device", "0")
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs") or 50)

    if not data_yaml.is_file():
        print(
            f"ERROR: data.yaml not found: {data_yaml}\n"
            "Run: python scripts/prepare_obb_dataset.py",
            file=sys.stderr,
        )
        return 1

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed.", file=sys.stderr)
        return 1

    cuda_ok = torch.cuda.is_available()
    if str(device) != "cpu" and not cuda_ok:
        print(
            "WARNING: CUDA is not available. "
            "Set --device cpu for a tiny smoke run, or train on a CUDA host.\n"
            f"  torch={torch.__version__} cuda_available={cuda_ok}"
        )
        if not args.dry_run and args.device is None:
            print(
                "ERROR: Refusing to start GPU training without CUDA. "
                "Re-run with --device cpu (smoke only) or on a GPU machine.",
                file=sys.stderr,
            )
            return 2

    print("Train config:", train_cfg_path)
    print("Data YAML:   ", data_yaml)
    print("Model:       ", model_name)
    print("Device:      ", device if cuda_ok or str(device) == "cpu" else f"{device} (CUDA missing)")
    print("Epochs:      ", epochs)
    print("CUDA:        ", cuda_ok)

    if args.dry_run:
        print("Dry run OK — not starting training.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. pip install ultralytics", file=sys.stderr)
        return 1

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=int(cfg.get("imgsz") or 640),
        batch=int(cfg.get("batch") or 8),
        device=device if (cuda_ok or str(device) == "cpu") else "cpu",
        workers=int(cfg.get("workers") or 4),
        project=str(resolve_path(cfg.get("project") or "models/runs")),
        name=str(cfg.get("name") or "obb_v1"),
        exist_ok=bool(cfg.get("exist_ok", True)),
        patience=int(cfg.get("patience") or 20),
        save=bool(cfg.get("save", True)),
        plots=bool(cfg.get("plots", True)),
    )

    # Copy best weights to models/your_obb.pt for platform inference
    best = Path(getattr(results, "save_dir", "")) / "weights" / "best.pt"
    if not best.is_file():
        # Ultralytics may expose path differently; try common location
        run_dir = resolve_path(cfg.get("project") or "models/runs") / str(cfg.get("name") or "obb_v1")
        best = run_dir / "weights" / "best.pt"

    dest = _REPO_ROOT / "models" / "your_obb.pt"
    if best.is_file():
        shutil.copy2(best, dest)
        print(f"Copied best weights → {dest}")
    else:
        print(f"WARNING: best.pt not found at {best}; check Ultralytics run directory.")

    # Quick val pass
    metrics = model.val(data=str(data_yaml), device=device if cuda_ok or str(device) == "cpu" else "cpu")
    print("Validation metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
