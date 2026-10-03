# Model card — obb_v1

Detector weights for Traffilytics. The file is `models/your_obb.pt`. It is gitignored; the Epic 7 release attaches it. The machine-readable record of the same run is [`models/provenance.json`](../models/provenance.json).

## Architecture

YOLOv8n-OBB, trained with Ultralytics from the pretrained checkpoint `yolov8n-obb.pt`. Task `obb`. Inference and evaluation resize inputs to 640.

## Training data

UAV-OBB, Mendeley Data V3, [doi:10.17632/6snrjwcpkh.3](https://doi.org/10.17632/6snrjwcpkh.3). The published images and labels were not modified. Training resized inputs to 640; that is a loader setting, not a change to the dataset files.

| Split | Images | Role |
|-------|--------|------|
| `train` | 1,158 | Training |
| `valid` | 207 | Headline metrics below |
| `test` | 10 | Spot-check only. Too small for a reported mAP (FR-DET-006). |

| Class id | Name |
|----------|------|
| 0 | bike |
| 1 | bus |
| 2 | car |
| 3 | other_vehicle |
| 4 | taxi |
| 5 | truck |

## Training

Config: `models/configs/train_obb.yaml`. The resolved run is `models/runs/obb_v1` (gitignored).

| Setting | Value |
|---------|-------|
| Epochs | 50 |
| Patience | 20 |
| Batch | 8 |
| Image size | 640 |
| Optimizer | auto |
| Seed | 0 |
| Deterministic | true |
| Pretrained init | `yolov8n-obb.pt` |
| Device | CUDA |

The Kaggle run did not record a git commit. The release tag that attaches `models/your_obb.pt` is the corresponding source.

## Evaluation

`python scripts/eval_obb.py --weights models/your_obb.pt` on the UAV-OBB `valid` split (Ultralytics split name `val`), 207 images. Confidence threshold 0.25, IoU threshold 0.7, image size 640. Stored in `models/runs/eval_obb/metrics.json`.

| Metric | Value |
|--------|-------|
| Precision | 0.854 |
| Recall | 0.853 |
| mAP50 | 0.838 |
| mAP50-95 | 0.688 |

Exact figures are in `models/provenance.json`. This eval file has no per-class mAP. These numbers are the product eval at confidence 0.25. The training log's validation scores use Ultralytics' default confidence cutoff and are higher; they are not the figures to quote.

## Licences

The weights are AGPL-3.0-only, the same as Traffilytics, because they were produced with Ultralytics YOLO. See [`LICENSE`](../LICENSE).

UAV-OBB is CC BY 4.0. Cite it wherever its imagery or results from it appear:

> Ahmad, Israr; Fengjun, Shang; Bibi, Kiran; Slaman Pathan, Muhammad (2026), *UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote Sensing Object Detection in Smart Cities*, Mendeley Data, V3, doi: [10.17632/6snrjwcpkh.3](https://doi.org/10.17632/6snrjwcpkh.3)

This card describes the detector only. Tracking and traffic counts are separate and are not measured here.
