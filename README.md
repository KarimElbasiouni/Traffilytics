# Traffilytics

AI Traffic Intelligence Platform: modular video ingestion, trained OBB detection, multi-object tracking, trajectory generation, analytics, APIs, and an interactive dashboard.

**UAV-OBB** ([Mendeley Data](https://data.mendeley.com/datasets/6snrjwcpkh/3), CC BY 4.0) is the training and evaluation dataset. At runtime Traffilytics processes **user-uploaded video**; the dataset's bundled MP4 clips serve as demo and sanity-check footage. UAV-OBB ships no ground-truth trajectories, so trajectories always come from Traffilytics' own detector and tracker.

Full design docs live under [`docs/`](docs/).

## Quick start

### 1. Conda environment

```bash
conda create -n traffilytics python=3.10 pip -y
conda activate traffilytics
pip install -e ".[dev]"
# Optional for Epic 2+ (CPU torch for local smoke; use CUDA builds on GPU hosts)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
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
```

No dataset credentials are required — the downloader uses Mendeley's public API.

### 3. Get the dataset (Epic 1)

```bash
python scripts/download_uav_obb.py
```

This downloads UAV-OBB (~506 MiB), verifies the SHA-256 that Mendeley publishes, then installs the annotated splits into `data/annotations/` and the bundled demo clips into `data/raw/`. Allow roughly 2.5 GB free on the data volume while it extracts.

| Flag | Effect |
|------|--------|
| `--dry-run` | Print file size, checksum, and destinations; download nothing |
| `--src PATH` | Install from a `UAV-OBB.zip` or extracted folder you already have (no network) |
| `--skip-videos` | Install only the annotated splits |
| `--keep-archive` | Keep the downloaded zip (deleted after extraction by default) |

**Do not** run this from pytest or CI (NFR-TEST-005); the test suite is fully offline.

### 4. Ingest a video (Epic 1)

Use one of the demo clips now sitting in `data/raw/`, or drop your own there:

```bash
python scripts/ingest_video.py --video data/raw/<clip>.mp4 --config configs/default.yaml
```

This writes metadata and frames under `data/processed/<video_id>/`. Frame stride defaults to 2 (`video.frame_stride`), leaving ~15 effective fps from a 30 fps source so tracking can associate vehicles between frames; raise it for throughput, lower it for quality. Pre-stabilized clips are preferred where available — Traffilytics does not implement stabilization.

### 5. Tests

```bash
pytest
```

Two CLI tests skip until `models/configs/uav_obb_data.yaml` exists; see [Write the Ultralytics YAML](#write-the-ultralytics-yaml).

## The dataset

| Spec | Value |
|------|-------|
| Content | 1,375 JPEG images at 1920×1080, 35,615 oriented vehicle instances |
| Source | UAV footage over urban roads in Chongqing and Wuhan, China, at 75–108 m |
| Conditions | Morning, midday, evening, rain, mist/light fog; wide FOV and zoom |
| Splits | `train/` 1,158 · `valid/` 207 · `test/` 10 images |
| Classes | bike (`0`), bus (`1`), car (`2`), other_vehicle (`3`), taxi (`4`), truck (`5`) |
| Labels | YOLOv8-OBB — class id plus four corner vertices in normalized coordinates |
| Video | `test_videos_mp4/` — one short clip with sparsely annotated reference frames, plus two longer unannotated sequences |
| Licence | CC BY 4.0 (attribution required) |
| Archive | 505.5 MiB zip, DOI [10.17632/6snrjwcpkh.3](https://doi.org/10.17632/6snrjwcpkh.3) |

The `test` split holds only **10 images**, far too few for a stable mAP, so headline detector metrics are reported on the 207-image `valid` split and `test` serves as a qualitative spot-check (FR-DET-006).

### Data layout

| Path | Role |
|------|------|
| `data/raw/` | Uploaded video plus UAV-OBB demo clips |
| `data/annotations/` | UAV-OBB `train` / `valid` / `test` splits and `data.yaml` |
| `data/processed/` | Ingest frames and `metadata.json`; Epic 2 `detections.json` |

Large media and `.pt` weights are gitignored, so a clone carries the folder skeleton only.

### Write the Ultralytics YAML

```bash
python scripts/prepare_obb_dataset.py
```

Scans `data/annotations/` and writes `models/configs/uav_obb_data.yaml` (gitignored) with an absolute `path:` and the six-class map. UAV-OBB ships its own `data.yaml`, but the generated file is preferred because its paths resolve against this checkout.

## Train, evaluate, and detect (Epic 2)

**YOLO OBB training needs a CUDA-capable host.** This environment may only have CPU PyTorch. Dry-runs, pytest, and CLI wiring work here; product inference (FR-DET-001) needs `models/your_obb.pt` from a GPU train.

Inference defaults (`conf_threshold`, `iou_threshold`, `imgsz`, `device`) live in [`configs/default.yaml`](configs/default.yaml) under `detection:`. Keep `detection.allow_pretrained: false`. Training hyperparameters live in [`models/configs/train_obb.yaml`](models/configs/train_obb.yaml), which defaults to the `yolov8n-obb.pt` nano checkpoint for smoke runs — override `model` with a larger OBB checkpoint such as `yolo11m-obb.pt` for a real GPU train. Label parsing lives in `computer_vision/detection/obb_labels.py`; dataset discovery and installation in `adapters/uav_obb/`.

### CUDA host — train → eval → detect

```bash
conda activate traffilytics
# 1) Install the dataset and write the data.yaml
python scripts/download_uav_obb.py
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
# Optional side-by-side comparison (metrics JSON only — never the product model)
python scripts/eval_obb.py --weights models/your_obb.pt --baseline path/to/other.pt
```

Writes `models/runs/eval_obb/metrics.json` (mAP50 / mAP50-95) plus qualitative overlays, defaulting to the `valid` split.

A `--baseline` is comparable only when its class map matches UAV-OBB's six classes. A DOTA-pretrained checkpoint such as `yolo11n-obb.pt` predicts a different taxonomy, so its mAP here is a wiring smoke test rather than a score to beat; your own earlier training runs are the meaningful baselines.

### Detect on ingested frames (Epic 3 input)

```bash
python scripts/detect_frames.py --video-id <video_id>
python scripts/detect_frames.py --video-id <video_id> --overlays
```

Writes `data/processed/<video_id>/detections.json` (FR-DET records). Optional overlays land in `data/processed/<video_id>/det_overlays/`.

### Track detections into trajectories (Epic 3)

```bash
python scripts/track_video.py --video-id <video_id>
python scripts/track_video.py --video-id <video_id> --overlays
python scripts/track_video.py --video-id <video_id> --lanes configs/lanes/<video_id>.json
python scripts/track_video.py --detections data/processed/<video_id>/detections.json --dry-run
```

Reads `detections.json` and writes `trajectories.json` plus `tracking_diagnostics.json`. `--overlays` draws `track_id` on ingested frames (`track_overlays/` stills and `tracks_overlay.mp4`). Pose fields stay the detector's OBBs.

Lane labels come from a per-video JSON under `configs/lanes/` (FR-TRK-006). If that file is missing, `lane` stays `null`. Example:

```json
{
  "video_id": "clip",
  "lanes": [{ "id": "lane_1", "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]] }],
  "zones": []
}
```

Diagnostics are **not** MOTA: UAV-OBB has no trajectory ground truth. `suspected_id_switches` counts same-class handoffs (a track ends and another starts nearby). Needs `ultralytics` (and `lap`) on the real path; `--dry-run` only checks the detections file.

### Analyze trajectories (Epic 4)

```bash
python scripts/analyze_video.py --video-id <video_id>
python scripts/analyze_video.py --video-id <video_id> --lanes configs/lanes/<video_id>.json
python scripts/analyze_video.py --trajectories data/processed/<video_id>/trajectories.json --dry-run
python scripts/analyze_video.py --video-id <video_id> --pixels-per-metre 12.5
```

Reads `trajectories.json` and writes `analytics.json` (flow, flow–density, bottleneck, imbalance, events, template insights). Speed and density stay **pixel-labelled** unless you pass `--pixels-per-metre` (or set `analytics.pixels_per_metre` in config). Lane imbalance needs stamped `lane` values or a lane JSON; bottleneck location needs `zones` in that JSON. Missing lanes/zones are skipped, not invented.

`--dry-run` only checks the trajectories file. Optional LC/TTC (`analytics/micro/`) is not implemented.

### API + dashboard (Epic 5)

The dashboard is a **Vite + React SPA**. FastAPI remains the only backend (no Next.js).

```bash
pip install -e ".[api]"
# Terminal 1 — API + worker
python scripts/run_api.py --host 127.0.0.1 --port 8000
# Terminal 2 — UI with hot reload (proxies /api to port 8000)
cd frontend && npm install && npm run dev
```

Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/). For a single-process demo, build the SPA and let FastAPI serve `frontend/dist`:

```bash
cd frontend && npm run build
python scripts/run_api.py --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). OpenAPI is at `/docs`. `POST /api/v1/videos` returns a `job_id` immediately; a worker runs ingest → detect → track → analytics → SQLite persist.

The default DB is `sqlite:///./data/traffilytics.db` (`DB_URL` in `.env`). Detection still needs `models/your_obb.pt`. To load CLI artifacts, copy `trajectories.json` into `data/processed/<video_id>/` and use **Re-process this clip** (or `POST .../process` with `{"reuse_artifacts": true}`).

Lane polygons are stored with `PUT /api/v1/videos/{video_id}/lanes` and written to `configs/lanes/<video_id>.json`.

## Licences and attribution

### Traffilytics — AGPL-3.0

Traffilytics is licensed **[AGPL-3.0-only](LICENSE)**.

This follows from Ultralytics YOLO, which drives OBB training and inference here and is itself AGPL-3.0. Ultralytics treats its code, architectures, training pipelines, **and the resulting trained weights** as covered, so using it without an Enterprise License requires publishing the entire project under AGPL-3.0 — which is what this project does (NFR-DOC-004). The AGPL's network clause (section 13) reaches hosted services, so a publicly deployed Traffilytics instance must offer its complete corresponding source, not just a downloadable copy.

### Trained weights

`models/your_obb.pt` is gitignored because git handles large binaries poorly, but it is **not** exempt from the above: publish it as a GitHub release or model repository alongside any public deployment. Weights derive from both an Ultralytics pretrained checkpoint (AGPL-3.0) and UAV-OBB imagery (CC BY 4.0), so a model card should carry both notices.

### UAV-OBB — CC BY 4.0

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which requires attribution wherever its imagery or derived results appear (NFR-DOC-003):

> Ahmad, Israr; Fengjun, Shang; Bibi, Kiran; Slaman Pathan, Muhammad (2026), *UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote Sensing Object Detection in Smart Cities*, Mendeley Data, V3, doi: [10.17632/6snrjwcpkh.3](https://doi.org/10.17632/6snrjwcpkh.3)

This section summarizes the applicable terms and is not legal advice.
