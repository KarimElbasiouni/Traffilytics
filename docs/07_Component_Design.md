# Component Design

Traffilytics components are **platform-owned**. DRIFT supplies data, annotation format, and GT trajectories for training/evaluation—not the application implementation.

---

## Component 1 — Video Processing Module

### Purpose

Modular ingestion and preparation of traffic video for training and inference.

### Responsibilities (Traffilytics)

- Load video files (prefer DRIFT stabilized clips when available)
- Extract frames (OpenCV)
- Normalize formats as needed
- Store and manage metadata (`video_id`, `site`, fps, resolution, duration)
- Organize raw vs processed artifacts

### Not a focus

- Reimplementing Stabilo / DRIFT stabilization R&D—reuse stabilized videos or their process if needed

### Low-Level Design

**Class:** `VideoProcessor`

| Method | Description |
|--------|-------------|
| `load_video()` | Open and validate video source |
| `extract_frames()` | Decode frames |
| `get_metadata()` | Return site, fps, resolution, duration, ids |
| `save_frames()` / artifact paths | Persist intermediates as configured |

### Requirements

FR-VID-001 … FR-VID-006

---

## Component 2 — Vehicle Detection Module

### Purpose

Train and run an OBB vehicle detector on DRIFT-format annotations.

### Technology

- **Architecture:** YOLO OBB (e.g., YOLOv11 OBB family)—not reinvented
- **Weights:** Traffilytics-trained on DRIFT annotations
- **Baseline (optional):** Compare against DRIFT-provided `best.pt` during evaluation

### Responsibilities

- Dataset adapters for DRIFT OBB labels
- Training loop / training entrypoint (PyTorch / Ultralytics-style tooling)
- Inference producing OBB + `class_id` + confidence
- Evaluation against held-out DRIFT annotations

### Supported Classes

| `class_id` | Label |
|------------|-------|
| 0 | Bus |
| 1 | Car |
| 2 | Truck |

### Low-Level Design

**Classes:** `VehicleDetector`, `DetectionTrainer`, `DetectionEvaluator`

| Method | Description |
|--------|-------------|
| `train()` | Train OBB model on DRIFT splits |
| `load_model()` | Load Traffilytics weights |
| `detect_objects()` | OBB inference on a frame |
| `evaluate()` | Metrics vs DRIFT annotations |

### Requirements

FR-DET-001 … FR-DET-006

---

## Component 3 — Tracking & Trajectory Generation

### Purpose

Independently integrate multi-object tracking and generate Traffilytics trajectories.

### Technology

- **Primary:** ByteTrack (integrate, do not reimplement the algorithm)
- **Optional comparison:** OC-SORT, DeepSORT

### Responsibilities

- Associate OBB detections across frames
- Assign `track_id`
- Emit generated trajectory streams/files
- Lane inference or utilization of lane cues for analytics/visualization
- Benchmark generated trajectories against DRIFT GT CSVs

### Explicit non-responsibility

- Using DRIFT GT trajectory CSVs as the live trajectory source for analytics/dashboard

### Low-Level Design

**Classes:** `VehicleTracker`, `TrajectoryGenerator`, `TrajectoryBenchmarker`

| Method | Description |
|--------|-------------|
| `initialize_tracker()` | Configure ByteTrack (or alternative) |
| `update_tracks()` | Ingest frame detections |
| `generate_trajectory()` | Export Traffilytics trajectories |
| `benchmark_against_gt()` | Compare to DRIFT GT CSVs |

### Requirements

FR-TRK-001 … FR-TRK-006

---

## Component 4 — Traffic Analytics Engine

### Purpose

Platform-specific analytics designed for Traffilytics—not a packaging of DRIFT example scripts.

### Feature modules

| Module | Traffilytics ownership |
|--------|------------------------|
| 4.1 Flow characterization | Own metrics/algorithms (volume, speed, density, state, flow–density) |
| 4.2 Bottleneck detection | Own zone methodology and cause attribution |
| 4.3 Flow imbalance | **Dedicated** lane utilization / imbalance feature |
| 4.4 Event detection | Rule-based: sudden congestion, stopped vehicle, queue spillback |
| 4.5 Optional micro | LC / TTC as platform features if prioritized |

### Requirements

FR-FLOW-*, FR-BTN-*, FR-IMB-*, FR-EVT-*, FR-MIC-*

---

## Component 5 — Automated Insight Generation

### Purpose

Convert analytics into human-readable summaries (**not present in DRIFT**).

### Low-Level Design

**Class:** `InsightGenerator`

| Method | Description |
|--------|-------------|
| `analyze_metrics()` | Select salient metrics |
| `identify_key_events()` | Rank / filter events |
| `generate_summary()` | Template (V1) or LLM (V2) text |

---

## Component 6 — Database Layer

### Purpose

Persistent store for videos, **generated** trajectories, analytics, and events (**not provided by DRIFT**).

See [08_Database_Design.md](./08_Database_Design.md).

---

## Component 7 — Backend API

### Purpose

Manage processing, storage, and data access (e.g., **FastAPI**).

### Responsibilities

- Register videos / jobs
- Trigger process pipeline
- Query trajectories, analytics, events, insights
- Optionally expose train/eval job status and benchmark results

---

## Component 8 — Dashboard & Reporting

### Purpose

Interactive web traffic intelligence UI and automated reports (**not in DRIFT**).

| Page | Displays |
|------|----------|
| Overview | Totals, traffic state, site, major events |
| Traffic Flow | Volume, speed, congestion, flow–density |
| Bottleneck | Heatmaps, locations, accumulation |
| Flow Imbalance | Lane utilization |
| Events | Events, timestamps, severity |
| Reports | Automated summaries and findings |

---

## Component Interaction

```
User → Dashboard → FastAPI
         → VideoProcessor
         → VehicleDetector (trained OBB weights)
         → VehicleTracker (ByteTrack)
         → TrajectoryGenerator ──► DB
         → (optional) TrajectoryBenchmarker vs DRIFT GT
         → Analytics Engine → InsightGenerator ──► DB
User ← Dashboard / Reports
```

### Training (offline / job)

```
DRIFT annotations → DetectionTrainer → models/your_obb.pt
                                      → DetectionEvaluator
```
