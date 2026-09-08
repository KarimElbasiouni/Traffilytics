# Traffilytics

AI Traffic Intelligence Platform: modular video ingestion, trained OBB detection, multi-object tracking, trajectory generation, analytics, APIs, and an interactive dashboard.

**DRIFT** ([Hj-Lee/The-DRIFT](https://huggingface.co/datasets/Hj-Lee/The-DRIFT)) is the primary training and evaluation dataset. Ground-truth trajectory CSVs are for **validation and benchmarking only** — live trajectories always come from Traffilytics’ detector + tracker.

Full design docs live under [`docs/`](docs/).

## Quick start

### 1. Conda environment

```bash
conda create -n traffilytics python=3.10 pip -y
conda activate traffilytics
pip install -e ".[dev]"
# Optional for Epic 2+ (CPU torch for local smoke; use CUDA builds on GPU hosts)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics huggingface_hub
```

Or from the full spec file (may take longer):

```bash
conda env create -f environment.yml
conda activate traffilytics
pip install -e .
```

### 2. Environment variables

```bash
cp .env.example .env
# Optionally set HF_TOKEN for Hugging Face downloads
```

### 3. Ingest a video (Epic 1)

Place a clip under `data/raw/` (or download a DRIFT sample — see below), then:

```bash
python scripts/ingest_video.py --video data/raw/<clip>.mp4 --config configs/default.yaml
```

This writes metadata and frames under `data/processed/<video_id>/`.

### 4. Tests

```bash
pytest
```

## Data layout

| Path | Role |
|------|------|
| `data/raw/` | DRIFT (or other) videos — prefer stabilized clips |
| `data/annotations/` | DRIFT OBB labels / splits |
| `data/gt_trajectories/` | DRIFT GT CSVs (**Epic 3 benchmark only** — unused for detection) |
| `data/processed/` | Ingest frames + `metadata.json`; Epic 2 `detections.json` |

Large media and `.pt` weights are gitignored.

## Download a DRIFT sample

```bash
# Requires network; set HF_TOKEN in .env if needed
python scripts/download_drift_sample.py
```

See script help for options. Prefer stabilized videos; Traffilytics does **not** focus on reimplementing Stabilo.

The sample downloader writes **two OBB frames** under `data/annotations/` for layout smoke only. That is not enough to train.

## Full DRIFT OBB annotations (Epic 2)

Real detector training needs the GitHub [`model/{train,valid,test}`](https://github.com/AIxMobility/The-DRIFT) splits:

| | |
|---|---|
| Size | ~2,301 frames / ~300K OBB instances (bus / car / truck) |
| Layout | `model/train`, `model/valid`, `model/test` (+ `data.yaml`) |
| Dest | `data/annotations/` — **gitignored**, do not commit |
| CI | **Do not** download this 4K set in pytest or CI |

Clone [The-DRIFT](https://github.com/AIxMobility/The-DRIFT) (or otherwise obtain its `model/` tree), then copy into Traffilytics and write the Ultralytics YAML:

```bash
# --src may be the repo root or …/The-DRIFT/model
python scripts/download_obb_dataset.py --full --src /path/to/The-DRIFT
python scripts/prepare_obb_dataset.py
```

`--dry-run` prints the copy plan without writing files. `--full` is required so a huge copy cannot happen by accident. The helper only copies a **local** tree; it does not clone GitHub.

Generated `models/configs/drift_obb_data.yaml` is gitignored.

## Train, evaluate, and detect (Epic 2)

**YOLO OBB training needs a CUDA-capable host.** This environment may only have CPU PyTorch. Dry-runs, pytest, and CLI wiring work here; product inference (FR-DET-001) needs `models/your_obb.pt` from a GPU train.

DRIFT ground-truth trajectory CSVs under `data/gt_trajectories/` stay **unused** in this step. They are for Epic 3 tracker benchmarking only — never as live trajectories.

Inference defaults (`conf_threshold`, `iou_threshold`, `imgsz`, `device`) live in [`configs/default.yaml`](configs/default.yaml) under `detection:`. Keep `detection.allow_pretrained: false`. Training hyperparameters: [`models/configs/train_obb.yaml`](models/configs/train_obb.yaml) (nano smoke checkpoint). On a GPU host, override `model` to `yolo11m-obb.pt` for a real train (DRIFT paper: YOLOv11m OBB, batch 4, IoU 0.7) — do not switch this CPU box to `m`. Annotation adapter: `adapters/drift/obb_annotations.py`.

### CUDA host — train → eval → detect

```bash
conda activate traffilytics
# 1) Place full DRIFT OBB splits under data/annotations/ (see above)
python scripts/download_obb_dataset.py --full --src /path/to/The-DRIFT
python scripts/prepare_obb_dataset.py

# 2) Train Traffilytics weights (copies best.pt → models/your_obb.pt)
python scripts/train_obb.py --train-config models/configs/train_obb.yaml

# 3) Held-out metrics + overlays
python scripts/eval_obb.py --weights models/your_obb.pt

# 4) Infer on ingested frames (Epic 3 input)
python scripts/ingest_video.py --video data/raw/<clip>.mp4
python scripts/detect_frames.py --video-id <video_id>
```

### CPU — path checks (no GPU, no weights required)

```bash
python scripts/train_obb.py --dry-run
python scripts/eval_obb.py --dry-run
python scripts/detect_frames.py --dry-run --video-id <video_id>
```

Missing `models/your_obb.pt` fails with a clear error. `--allow-pretrained` loads `yolo11n-obb.pt` with a loud warning — **CPU wiring only**, not FR-DET-001 compliant.

### Evaluate held-out labels

```bash
python scripts/eval_obb.py --weights models/your_obb.pt
# Optional: DRIFT best.pt comparison (metrics JSON only — never the product model)
python scripts/eval_obb.py --weights models/your_obb.pt --baseline path/to/best.pt
```

Writes `models/runs/eval_obb/metrics.json` (mAP50 / mAP50-95) plus qualitative overlays.

### Detect on ingested frames (Epic 3 input)

```bash
python scripts/detect_frames.py --video-id <video_id>
python scripts/detect_frames.py --video-id <video_id> --overlays
```

Writes `data/processed/<video_id>/detections.json` (FR-DET records). Optional overlays: `data/processed/<video_id>/det_overlays/`.

## Attribution

Cite DRIFT when publishing results: [arXiv:2504.11019](https://arxiv.org/abs/2504.11019). Acknowledge Stabilo only if its stabilization code/process is used.
