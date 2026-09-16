# Component Design

Traffilytics components are **platform-owned**. UAV-OBB supplies training images, OBB labels, and the annotation format — not the application implementation.

---

## Component 1 — Video Processing Module

### Purpose

Modular ingestion and preparation of traffic video for training and inference.

### Responsibilities (Traffilytics)

- Load user-uploaded video files, plus UAV-OBB's bundled MP4 clips for demos
- Extract frames (OpenCV), with configurable downscaling and frame stride
- Normalize formats as needed
- Store and manage metadata (`video_id`, `site` label, fps, resolution, duration)
- Organize raw vs processed artifacts

### Not a focus

- Video stabilization. Clips are processed as supplied.

### Low-Level Design

**Class:** `VideoProcessor`

| Method | Description |
|--------|-------------|
| `load_video()` | Open and validate video source |
| `extract_frames()` | Decode frames |
| `get_metadata()` | Return scene label, fps, resolution, duration, ids |
| `save_frames()` / artifact paths | Persist intermediates as configured |

### Requirements

FR-VID-001 … FR-VID-006

---

## Component 2 — Vehicle Detection Module

### Purpose

Train and run an OBB vehicle detector on UAV-OBB annotations.

### Technology

- **Architecture:** YOLO OBB (YOLOv8/YOLOv11 OBB family) — not reinvented
- **Weights:** Traffilytics-trained on UAV-OBB splits
- **Baseline (optional):** Compare against a public pretrained OBB checkpoint during evaluation, without adopting it as the product model

### Responsibilities

- Dataset adapter for UAV-OBB OBB labels and `data.yaml`
- Training loop / training entrypoint (PyTorch / Ultralytics tooling)
- Inference producing OBB + `class_id` + confidence
- Evaluation against held-out UAV-OBB annotations

### Supported Classes

Read from the dataset `data.yaml` where practical; hard-coded maps must match it.

| `class_id` | Label |
|------------|-------|
| 0 | bike |
| 1 | bus |
| 2 | car |
| 3 | other_vehicle |
| 4 | taxi |
| 5 | truck |

### Low-Level Design

**Classes:** `VehicleDetector`, `DetectionTrainer`, `DetectionEvaluator`

| Method | Description |
|--------|-------------|
| `train()` | Train OBB model on UAV-OBB splits |
| `load_model()` | Load Traffilytics weights |
| `detect_objects()` | OBB inference on a frame |
| `evaluate()` | Metrics vs held-out UAV-OBB annotations |

### Requirements

FR-DET-001 … FR-DET-006

---

## Component 3 — Tracking & Trajectory Generation

### Purpose

Independently integrate multi-object tracking and generate Traffilytics trajectories. This component supplies the temporal dimension that the dataset lacks.

### Technology

- **Primary:** ByteTrack (integrate, do not reimplement the algorithm)
- **Optional comparison:** OC-SORT, DeepSORT

### Responsibilities

- Associate OBB detections across frames
- Assign `track_id`
- Emit generated trajectory streams/files
- Resolve lane membership from user-defined lane polygons
- Report tracking stability diagnostics (no trajectory ground truth exists)

### Explicit non-responsibility

- Producing or importing a ground-truth trajectory dataset

### Low-Level Design

**Classes:** `VehicleTracker`, `TrajectoryGenerator`, `LaneAssigner`, `TrackingDiagnostics`

| Method | Description |
|--------|-------------|
| `initialize_tracker()` | Configure ByteTrack (or alternative) |
| `update_tracks()` | Ingest frame detections |
| `generate_trajectory()` | Export Traffilytics trajectories |
| `assign_lane()` | Map a trajectory point to a configured lane polygon |
| `track_stability_report()` | Track counts, ID switches, fragmentation, overlay video |

### Requirements

FR-TRK-001 … FR-TRK-006

---

## Component 4 — Traffic Analytics Engine

### Purpose

Platform-specific analytics designed for Traffilytics.

### Feature modules

| Module | Traffilytics ownership |
|--------|------------------------|
| 4.1 Flow characterization | Own metrics/algorithms (volume, speed, density, state, flow–density) |
| 4.2 Bottleneck detection | Own zone methodology and cause attribution |
| 4.3 Flow imbalance | **Dedicated** lane utilization / imbalance feature over configured lane polygons |
| 4.4 Event detection | Rule-based: sudden congestion, stopped vehicle, queue spillback |
| 4.5 Optional micro | LC / TTC as platform features if prioritized |

### Units

Speed and density are computed in pixel space by default. When a per-video pixel-to-metre scale is configured, values are converted and labelled as physical units; otherwise they stay pixel-based and are labelled as such.

### Requirements

FR-FLOW-*, FR-BTN-*, FR-IMB-*, FR-EVT-*, FR-MIC-*

---

## Component 5 — Automated Insight Generation

### Purpose

Convert analytics into human-readable summaries.

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

Persistent store for videos, **generated** trajectories, analytics, and events.

See [08_Database_Design.md](./08_Database_Design.md).

---

## Component 7 — Backend API

### Purpose

Manage uploads, processing, storage, and data access (e.g., **FastAPI**).

### Responsibilities

- Accept video uploads and register them
- Enqueue processing jobs and return a job id without blocking
- Expose job status and progress
- Query trajectories, analytics, events, insights
- Expose training/evaluation job status and detector metrics

---

## Component 8 — Dashboard & Reporting

### Purpose

Interactive web traffic intelligence UI and automated reports.

| Page | Displays |
|------|----------|
| Upload / Jobs | Upload a clip, watch job status |
| Overview | Totals, traffic state, scene label, major events |
| Traffic Flow | Volume, speed, congestion, flow–density |
| Bottleneck | Heatmaps, locations, accumulation |
| Flow Imbalance | Lane utilization |
| Events | Events, timestamps, severity |
| Reports | Automated summaries and findings |

---

## Component Interaction

```
User → Dashboard → FastAPI  (upload returns job id)
         → Worker job:
             → VideoProcessor
             → VehicleDetector (trained OBB weights)
             → VehicleTracker (ByteTrack)
             → TrajectoryGenerator + LaneAssigner ──► DB
             → (optional) TrackingDiagnostics
             → Analytics Engine → InsightGenerator ──► DB
User ← Dashboard / Reports
```

### Training (offline / job)

```
UAV-OBB splits → DetectionTrainer → models/your_obb.pt
                                   → DetectionEvaluator → metrics + overlays
```
