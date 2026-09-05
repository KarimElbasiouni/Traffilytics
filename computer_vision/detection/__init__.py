"""Detection package (Epic 2+): OBB types, training, evaluation, and inference."""

from computer_vision.detection.detector import DetectorError, VehicleDetector
from computer_vision.detection.evaluator import (
    DetectionEvaluator,
    EvalPlan,
    EvalResult,
    EvaluatorError,
    draw_obb_overlay,
    write_metrics_json,
    write_obb_overlay,
)
from computer_vision.detection.trainer import (
    CudaUnavailableError,
    DetectionTrainer,
    TrainPlan,
    TrainResult,
    TrainerError,
)
from computer_vision.detection.types import (
    CLASS_NAMES,
    Detection,
    as_corners,
    corners_to_cxcywhr,
    cxcywhr_to_corners,
)

__all__ = [
    "CLASS_NAMES",
    "CudaUnavailableError",
    "Detection",
    "DetectionEvaluator",
    "DetectionTrainer",
    "DetectorError",
    "EvalPlan",
    "EvalResult",
    "EvaluatorError",
    "TrainPlan",
    "TrainResult",
    "TrainerError",
    "VehicleDetector",
    "as_corners",
    "corners_to_cxcywhr",
    "cxcywhr_to_corners",
    "draw_obb_overlay",
    "write_metrics_json",
    "write_obb_overlay",
]
