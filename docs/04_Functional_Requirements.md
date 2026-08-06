# Functional Requirements

Requirements are grouped by system component. Traffilytics **owns** the pipeline; DRIFT supplies training/evaluation data and GT trajectories for benchmarking.

---

## 1. Video Processing Module (Traffilytics-owned)

| ID | Requirement |
|----|-------------|
| FR-VID-001 | The system shall accept DRIFT traffic video files (4K drone MP4 or equivalent). |
| FR-VID-002 | The system shall extract frames via Traffilytics’ own preprocessing pipeline (OpenCV or equivalent). |
| FR-VID-003 | The system shall store video metadata including `video_id`, `site`, `fps`, `resolution`, and `duration`. |
| FR-VID-004 | The system shall manage ingest paths, processed artifacts, and metadata as a modular pipeline (not a thin wrap of DRIFT scripts). |
| FR-VID-005 | The system shall support loading DRIFT media from Hugging Face and/or local paths for training and inference. |
| FR-VID-006 | The system may consume already-stabilized DRIFT videos; implementing stabilization is **not** a required focus. |

**Example metadata output**

```json
{
  "video_id": "site_03_clip_01",
  "site": "03",
  "fps": 30,
  "resolution": "3840x2160",
  "duration": 900,
  "source": "drift",
  "stabilized": true
}
```

---

## 2. Vehicle Detection Module

| ID | Requirement |
|----|-------------|
| FR-DET-001 | The system shall detect vehicles in video frames using a **Traffilytics-trained** YOLO OBB model. |
| FR-DET-002 | The system shall classify detected vehicles as bus, car, or truck (`class_id` 0/1/2). |
| FR-DET-003 | The system shall output polygon-based oriented bounding boxes (center/size/angle and/or four corners) compatible with DRIFT’s annotation format. |
| FR-DET-004 | The system shall output detection confidence scores in [0, 1]. |
| FR-DET-005 | The system shall **train** the OBB detector on DRIFT annotations (train/val splits). |
| FR-DET-006 | The system shall **evaluate** detector performance on held-out DRIFT annotations (e.g., mAP / qualitative overlays). |

**Note:** Using the YOLO architecture and optionally referencing DRIFT’s `best.pt` as a baseline is acceptable; MVP detection for the platform shall be driven by **your trained** weights.

**Example detection output**

```json
{
  "frame": 1204,
  "class_id": 1,
  "class": "car",
  "confidence": 0.94,
  "center_x": 160,
  "center_y": 230,
  "width": 80,
  "height": 60,
  "angle": 0.42,
  "corners": [[120,200],[200,200],[200,260],[120,260]]
}
```

---

## 3. Vehicle Tracking & Trajectory Generation

| ID | Requirement |
|----|-------------|
| FR-TRK-001 | The system shall maintain vehicle identity across frames via `track_id` using an independently integrated tracker. |
| FR-TRK-002 | The primary tracker shall be ByteTrack; the system may compare alternatives (OC-SORT, DeepSORT). |
| FR-TRK-003 | The system shall **generate** trajectories from Traffilytics detections + tracks (not by replaying DRIFT GT CSVs as the live source). |
| FR-TRK-004 | Generated trajectories shall include time-ordered pose fields suitable for analytics (center, size, angle/corners, class, confidence). |
| FR-TRK-005 | The system shall support **benchmarking** generated trajectories against DRIFT ground-truth trajectory CSVs. |
| FR-TRK-006 | The system shall infer or utilize lane information for analytics and visualization where available. |

**Example generated trajectory point**

```json
{
  "track_id": 52,
  "frame": 1204,
  "center_x": 145,
  "center_y": 320,
  "width": 78,
  "height": 58,
  "angle": 0.41,
  "class_id": 1,
  "confidence": 0.93,
  "site": "03",
  "lane": 2
}
```

---

## 4. Traffic Analytics Engine (Traffilytics-designed)

### 4.1 Traffic Flow Characterization

| ID | Requirement |
|----|-------------|
| FR-FLOW-001 | The system shall implement its own volume metrics (e.g., vehicles per minute) from generated trajectories. |
| FR-FLOW-002 | The system shall compute average speed and traffic density using platform-defined algorithms. |
| FR-FLOW-003 | The system shall classify congestion / traffic state over time windows. |
| FR-FLOW-004 | The system shall implement flow–density (or equivalent) characterization as a first-class analytics feature. |

### 4.2 Bottleneck Identification

| ID | Requirement |
|----|-------------|
| FR-BTN-001 | The system shall define analysis zones/regions for a scene/site. |
| FR-BTN-002 | The system shall analyze speed reduction, accumulation, and queue formation per zone using Traffilytics methodology. |
| FR-BTN-003 | The system shall identify primary bottleneck location(s) and likely cause. |
| FR-BTN-004 | The system should support heatmap-style views for bottleneck exploration. |

### 4.3 Traffic Flow Imbalance Analysis

| ID | Requirement |
|----|-------------|
| FR-IMB-001 | The system shall implement **dedicated** lane utilization / imbalance analysis. |
| FR-IMB-002 | The system shall compute direction or concentration metrics where heading/lane topology allows. |

### 4.4 Traffic Event Detection

| ID | Requirement |
|----|-------------|
| FR-EVT-001 | The system shall implement rule-based detection of sudden congestion. |
| FR-EVT-002 | The system shall implement rule-based detection of stopped vehicles. |
| FR-EVT-003 | The system shall implement rule-based detection of queue spillback. |
| FR-EVT-004 | The system shall emit event records with type, time, location/site, and severity. |

**Example event**

```json
{
  "type": "queue_spillback",
  "time": "00:14:05",
  "site": "03",
  "location": "East Approach",
  "severity": "high"
}
```

### 4.5 Optional Micro Metrics

| ID | Requirement |
|----|-------------|
| FR-MIC-001 | The system may implement lane-change detection as a platform analytics feature. |
| FR-MIC-002 | The system may implement TTC estimates as a platform analytics feature. |

These are Traffilytics features inspired by traffic-analysis needs—not a requirement to reuse DRIFT example scripts as the product implementation.

---

## 5. Automated Insight Generation

| ID | Requirement |
|----|-------------|
| FR-INS-001 | The system shall transform analytics results into human-readable summaries. |
| FR-INS-002 | The system shall highlight key congestion locations/sites, causes, and lane utilization imbalances. |

**Implementation path:** Version 1 rule-based templates; Version 2 optional LLM integration.

---

## 6. Database Layer

| ID | Requirement |
|----|-------------|
| FR-DB-001 | The system shall store video/site metadata. |
| FR-DB-002 | The system shall store vehicle/track records from **generated** trajectories. |
| FR-DB-003 | The system shall store trajectory points produced by Traffilytics. |
| FR-DB-004 | The system shall store time-series analytics. |
| FR-DB-005 | The system shall store detected events. |
| FR-DB-006 | The system may store evaluation/benchmark artifacts (e.g., metrics vs DRIFT GT) separately from live trajectory tables. |

---

## 7. Backend API

| ID | Requirement |
|----|-------------|
| FR-API-001 | The system shall expose a backend service (e.g., FastAPI) for processing, storage, and data access. |
| FR-API-002 | The API shall support triggering ingest/process jobs and querying videos, trajectories, analytics, events, and insights. |

---

## 8. Dashboard & Reporting

| ID | Requirement |
|----|-------------|
| FR-UI-001 | Overview page shall display total vehicles, traffic state, site context, and major events. |
| FR-UI-002 | Traffic Flow page shall display volume, speed trends, congestion timeline, and flow–density views. |
| FR-UI-003 | Bottleneck page shall display heatmaps, congestion locations, and accumulation. |
| FR-UI-004 | Flow Imbalance page shall display lane utilization (and direction distribution when available). |
| FR-UI-005 | Events page shall display detected events, timestamps, site/location, and severity. |
| FR-UI-006 | Reports page shall display automated summaries and key findings. |
| FR-UI-007 | Users shall be able to select a DRIFT site/clip and run Traffilytics processing. |
| FR-RPT-001 | The system shall generate automated reports/summaries for transportation analysis. |

---

## 9. Training, Evaluation & Deployment

| ID | Requirement |
|----|-------------|
| FR-ML-001 | Training pipelines shall use PyTorch / YOLO training tooling on DRIFT annotation splits. |
| FR-ML-002 | Evaluation pipelines shall report detection metrics and trajectory/tracking benchmarks vs DRIFT GT where applicable. |
| FR-DEP-001 | The application shall be structured as modular, deployable services (e.g., Dockerized). |

---

## 10. End-to-End MVP

| ID | Requirement |
|----|-------------|
| FR-MVP-001 | A complete run from DRIFT video → Traffilytics detect/track/trajectories → analytics → DB/API → dashboard/reports shall be achievable for at least one site. |
