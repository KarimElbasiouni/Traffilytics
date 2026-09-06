#!/usr/bin/env python3
"""Evaluate Traffilytics YOLO OBB weights on held-out DRIFT labels (Epic 2).

Thin CLI around :class:`computer_vision.detection.evaluator.DetectionEvaluator`.
Writes ``models/runs/eval_<name>/metrics.json`` plus qualitative overlays.
Optional ``--baseline`` compares a DRIFT ``best.pt`` in that JSON only — it is
never used as the product model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from computer_vision.detection.evaluator import (
    DetectionEvaluator,
    EvalPlan,
    EvalResult,
    EvaluatorError,
)
from computer_vision.preprocessing.config import load_config


def _detection_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read ``detection.*`` (falling back to ``models.weights``) from YAML config."""
    det = cfg.get("detection") or {}
    models = cfg.get("models") or {}
    return {
        "weights": det.get("weights") or models.get("weights") or "models/your_obb.pt",
        "conf_threshold": float(det.get("conf_threshold", 0.25)),
        "iou_threshold": float(det.get("iou_threshold", 0.7)),
        "imgsz": int(det.get("imgsz", 640)),
        "device": str(det.get("device") or "auto"),
        "allow_pretrained": bool(det.get("allow_pretrained", False)),
    }


def _print_plan(plan: EvalPlan) -> None:
    """Log resolved eval paths and settings (dry-run and real runs)."""
    print("Weights:     ", plan.weights_arg)
    if plan.weights_missing:
        print("              (missing — real eval needs --allow-pretrained or your_obb.pt)")
    print("Data YAML:   ", plan.data_yaml)
    print("Split:       ", plan.split)
    print("Device:      ", plan.device)
    print("Run dir:     ", plan.run_dir)
    print("Metrics:     ", plan.metrics_path)
    print("Overlays:    ", plan.overlay_dir)
    if plan.baseline is not None:
        status = "ok" if plan.baseline.is_file() else "missing"
        print(f"Baseline:     {plan.baseline} ({status})")
    if plan.pretrained_warning:
        print(f"WARNING: {plan.pretrained_warning}")


def _print_result(result: EvalResult) -> None:
    """Print plan, dry-run OK, or written metrics / baseline comparison."""
    _print_plan(result.plan)
    if result.dry_run:
        print("Dry run OK — not starting evaluation.")
        return
    if result.metrics_path is not None:
        print(f"Wrote metrics → {result.metrics_path}")
    metrics = result.metrics
    if "mAP50" in metrics:
        print("mAP50:       ", metrics["mAP50"])
    if "mAP50-95" in metrics:
        print("mAP50-95:    ", metrics["mAP50-95"])
    if result.overlay_dir is not None:
        print(f"Overlays:     {len(result.overlay_paths)} → {result.overlay_dir}")
    if result.baseline_metrics is not None:
        base_map = result.baseline_metrics.get("mAP50")
        print("Baseline mAP50:", base_map)
        print("Baseline is comparison-only (not the product model).")


def main(argv: list[str] | None = None) -> int:
    """Validate dataset YAML, optionally run val + overlays, write metrics.json."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO OBB weights on held-out DRIFT labels"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config for detection defaults (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Traffilytics weights (default: detection.weights / models/your_obb.pt)",
    )
    parser.add_argument(
        "--data",
        default="models/configs/drift_obb_data.yaml",
        help="Ultralytics data.yaml (default: models/configs/drift_obb_data.yaml)",
    )
    parser.add_argument(
        "--name",
        default="obb",
        help="Eval run name (writes models/runs/eval_<name>/)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Parent of the eval run dir (default: models/runs)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (auto, cpu, 0). Default from config.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (default from config)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold (default from config)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference / val image size (default from config)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Held-out split key in data.yaml (default: val)",
    )
    parser.add_argument(
        "--max-overlays",
        type=int,
        default=32,
        help="Max qualitative overlay images (default: 32)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional DRIFT best.pt for side-by-side metrics (comparison only)",
    )
    parser.add_argument(
        "--allow-pretrained",
        action="store_true",
        help="CPU wiring only: fall back to yolo11n-obb.pt (not FR-DET-001)",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip qualitative overlay images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data.yaml (and note missing weights) without val()",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    defaults = _detection_defaults(cfg)
    try:
        evaluator = DetectionEvaluator(
            args.weights or defaults["weights"],
            data=args.data,
            name=args.name,
            project=args.project,
            conf_threshold=(
                args.conf if args.conf is not None else defaults["conf_threshold"]
            ),
            iou_threshold=(
                args.iou if args.iou is not None else defaults["iou_threshold"]
            ),
            imgsz=args.imgsz if args.imgsz is not None else defaults["imgsz"],
            device=args.device if args.device is not None else defaults["device"],
            split=args.split,
            max_overlays=0 if args.no_overlays else args.max_overlays,
            allow_pretrained=args.allow_pretrained or defaults["allow_pretrained"],
            baseline=args.baseline,
        )
        result = evaluator.evaluate(
            dry_run=args.dry_run,
            write_overlays=not args.no_overlays,
        )
    except EvaluatorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
