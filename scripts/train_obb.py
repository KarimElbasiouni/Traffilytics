#!/usr/bin/env python3
"""Train Traffilytics YOLO OBB weights on DRIFT annotations (Epic 2).

Thin CLI around :class:`computer_vision.detection.trainer.DetectionTrainer`.
Requires a CUDA-capable GPU for practical training. CPU may run for smoke tests
with tiny datasets but is not recommended for full DRIFT training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from computer_vision.detection.trainer import (
    CudaUnavailableError,
    DetectionTrainer,
    TrainPlan,
    TrainResult,
    TrainerError,
)


def _print_plan(plan: TrainPlan) -> None:
    """Log resolved train paths and hyperparameters (dry-run and real runs)."""
    print("Train config:", plan.train_config_path or "(inline)")
    print("Data YAML:   ", plan.data_yaml)
    print("Model:       ", plan.model_name)
    print("Device:      ", plan.device_display)
    print("Epochs:      ", plan.epochs)
    print("CUDA:        ", plan.cuda_available)


def _print_result(result: TrainResult) -> None:
    """Print CUDA warning, plan, dry-run OK, or weight-copy / val metrics."""
    if result.plan.cuda_warning:
        print(f"WARNING: {result.plan.cuda_warning}")
    _print_plan(result.plan)
    if result.dry_run:
        print("Dry run OK — not starting training.")
        return
    if result.product_weights is not None:
        print(f"Copied best weights → {result.product_weights}")
    else:
        print(
            f"WARNING: best.pt not found at {result.plan.best_pt}; "
            "check Ultralytics run directory."
        )
    print("Validation metrics:", result.metrics)


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

    try:
        trainer = DetectionTrainer(
            args.train_config,
            data=args.data,
            device=args.device,
            epochs=args.epochs,
        )
        result = trainer.train(dry_run=args.dry_run)
    except CudaUnavailableError as exc:
        if exc.warning:
            print(f"WARNING: {exc.warning}")
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except TrainerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
