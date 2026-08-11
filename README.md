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
| `data/gt_trajectories/` | DRIFT GT CSVs (**eval only**) |
| `data/processed/` | Frames + `metadata.json` from Traffilytics ingest |

Large media and `.pt` weights are gitignored.

## Download a DRIFT sample

```bash
# Requires network; set HF_TOKEN in .env if needed
python scripts/download_drift_sample.py
```

See script help for options. Prefer stabilized videos; Traffilytics does **not** focus on reimplementing Stabilo.

## GPU / training note (Epic 2)

This environment may only have CPU PyTorch. **YOLO OBB training needs a CUDA-capable host.**

```bash
# 1) Download annotation samples (or place full DRIFT OBB splits under data/annotations/)
python scripts/download_drift_sample.py

# 2) Generate Ultralytics data.yaml
python scripts/prepare_obb_dataset.py

# 3) Dry-run path checks
python scripts/train_obb.py --dry-run

# 4) On a CUDA machine — train Traffilytics weights
python scripts/train_obb.py --train-config models/configs/train_obb.yaml
# Best weights are copied to models/your_obb.pt
```

Training config: [`models/configs/train_obb.yaml`](models/configs/train_obb.yaml). Annotation adapter: `adapters/drift/obb_annotations.py`.

## Attribution

Cite DRIFT when publishing results: [arXiv:2504.11019](https://arxiv.org/abs/2504.11019). Acknowledge Stabilo only if its stabilization code/process is used.
