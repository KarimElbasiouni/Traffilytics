# Functional Requirements

Requirements are grouped by system component. Traffilytics **owns** the pipeline; **UAV-OBB** supplies oriented-bounding-box images and labels for detector training and evaluation. Runtime footage comes from **user-uploaded video** (and UAV-OBB's bundled MP4 clips for demos).

Traffilytics does not consume any ground-truth trajectory dataset. All trajectories are **generated** by the platform's own detector and tracker.

---

## 1. Video Processing Module (Traffilytics-owned)

| ID | Requirement |
|----|-------------|
| FR-VID-001 | The system shall accept user-supplied aerial/drone traffic video files (MP4 or equivalent container), at resolutions from HD up to 4K. |
| FR-VID-002 | The system shall extract frames via Traffilytics’ own preprocessing pipeline (OpenCV or equivalent). |
| FR-VID-003 | The system shall store video metadata including `video_id`, `site`, `fps`, `resolution`, and `duration`, where `site` is an optional user-supplied scene label. |
| FR-VID-004 | The system shall manage ingest paths, processed artifacts, and metadata as a modular pipeline. |
| FR-VID-005 | The system shall load video from user upload and/or local paths; no external dataset service is required at runtime. |
| FR-VID-006 | The system may accept already-stabilized video; implementing stabilization is **not** a required deliverable. |

**Example metadata output**

```json
{
  "video_id": "uav_clip_01",
  "site": "chongqing_arterial",
  "fps": 30,
  "resolution": "1920x1080",
  "duration": 45,
  "source": "upload",
  "stabilized": false
}
```

---

## 2. Vehicle Detection Module

| ID | Requirement |
|----|-------------|
| FR-DET-001 | The system shall detect vehicles in video frames using a **Traffilytics-trained** YOLO OBB model. |
| FR-DET-002 | The system shall classify detected vehicles into the six UAV-OBB classes (`class_id` 0–5). |
| FR-DET-003 | The system shall output polygon-based oriented bounding boxes (four corners and/or center/size/angle) in the YOLO-OBB convention used by UAV-OBB labels. |
| FR-DET-004 | The system shall output detection confidence scores in [0, 1]. |
| FR-DET-005 | The system shall **train** the OBB detector on the UAV-OBB `train` / `valid` splits. |
| FR-DET-006 | The system shall **evaluate** detector performance on the held-out UAV-OBB `valid` split, reporting mAP50, mAP50-95, precision, and recall plus qualitative overlays. The 10-image `test` split is a spot-check only; headline metrics shall not be quoted from it. |

**Note:** Reusing the YOLO OBB architecture is intended. A publicly pretrained OBB checkpoint may be evaluated as a **baseline** for comparison, but platform detection shall be driven by **Traffilytics-trained** weights.

### Supported Classes

Class ids follow the UAV-OBB `data.yaml`. The shipped `data.yaml` is the source of truth and shall be read rather than hard-coded where practical.

| `class_id` | Label |
|------------|-------|
| 0 | bike |
| 1 | bus |
| 2 | car |
| 3 | other_vehicle |
| 4 | taxi |
| 5 | truck |

**Example detection output**

```json
{
  "frame": 1204,
  "class_id": 2,
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
| FR-TRK-003 | The system shall **generate** trajectories from Traffilytics detections + tracks; no external trajectory dataset shall be replayed as platform output. |
| FR-TRK-004 | Generated trajectories shall include time-ordered pose fields suitable for analytics (center, size, angle/corners, class, confidence). |
| FR-TRK-005 | The system shall assess tracking quality without trajectory ground truth — e.g. track counts, ID switches, track fragmentation, and overlay video — using UAV-OBB's sparsely annotated reference clip where applicable. |
| FR-TRK-006 | The system shall support **user-defined** lane and zone polygons per video, since no lane annotations accompany the dataset. |

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
  "class_id": 2,
  "confidence": 0.93,
  "video_id": "uav_clip_01",
  "lane": "lane_2"
}
```

`lane` is resolved from the user-defined lane polygons (FR-TRK-006), not from dataset metadata.

---

## 4. Traffic Analytics Engine (Traffilytics-designed)

### 4.1 Traffic Flow Characterization

| ID | Requirement |
|----|-------------|
| FR-FLOW-001 | The system shall implement its own volume metrics (e.g., vehicles per minute) from generated trajectories. |
| FR-FLOW-002 | The system shall compute average speed and traffic density using platform-defined algorithms. |
| FR-FLOW-003 | The system shall classify congestion / traffic state over time windows. |
| FR-FLOW-004 | The system shall implement flow–density (or equivalent) characterization as a first-class analytics feature. |
| FR-FLOW-005 | The system shall accept an optional pixel-to-metre scale per video; when absent, speed and density shall be reported in pixel-based units and labelled as such. |

### 4.2 Bottleneck Identification

| ID | Requirement |
|----|-------------|
| FR-BTN-001 | The system shall define analysis zones/regions per video scene. |
| FR-BTN-002 | The system shall analyze speed reduction, accumulation, and queue formation per zone using Traffilytics methodology. |
| FR-BTN-003 | The system shall identify primary bottleneck location(s) and likely cause. |
| FR-BTN-004 | The system should support heatmap-style views for bottleneck exploration. |

### 4.3 Traffic Flow Imbalance Analysis

| ID | Requirement |
|----|-------------|
| FR-IMB-001 | The system shall implement **dedicated** lane utilization / imbalance analysis over the user-defined lane polygons. |
| FR-IMB-002 | The system shall compute direction or concentration metrics from track heading where lane topology allows. |

### 4.4 Traffic Event Detection

| ID | Requirement |
|----|-------------|
| FR-EVT-001 | The system shall implement rule-based detection of sudden congestion. |
| FR-EVT-002 | The system shall implement rule-based detection of stopped vehicles. |
| FR-EVT-003 | The system shall implement rule-based detection of queue spillback. |
| FR-EVT-004 | The system shall emit event records with type, time, video/zone location, and severity. |

**Example event**

```json
{
  "type": "queue_spillback",
  "time": "00:14:05",
  "video_id": "uav_clip_01",
  "location": "zone_east_approach",
  "severity": "high"
}
```

### 4.5 Optional Micro Metrics

| ID | Requirement |
|----|-------------|
| FR-MIC-001 | The system may implement lane-change detection as a platform analytics feature. |
| FR-MIC-002 | The system may implement TTC estimates as a platform analytics feature. |

These are Traffilytics-owned features; they are not required to mirror any third-party research script.

---

## 5. Automated Insight Generation

| ID | Requirement |
|----|-------------|
| FR-INS-001 | The system shall transform analytics results into human-readable summaries. |
| FR-INS-002 | The system shall highlight key congestion zones, causes, and lane utilization imbalances. |

**Implementation path:** Version 1 rule-based templates; Version 2 optional LLM integration.

---

## 6. Database Layer

| ID | Requirement |
|----|-------------|
| FR-DB-001 | The system shall store video/scene metadata. |
| FR-DB-002 | The system shall store vehicle/track records from **generated** trajectories. |
| FR-DB-003 | The system shall store trajectory points produced by Traffilytics. |
| FR-DB-004 | The system shall store time-series analytics. |
| FR-DB-005 | The system shall store detected events. |
| FR-DB-006 | The system may store model evaluation artifacts (detection metrics, tracking diagnostics) separately from live trajectory tables. |

---

## 7. Backend API

| ID | Requirement |
|----|-------------|
| FR-API-001 | The system shall expose a backend service (e.g., FastAPI) for processing, storage, and data access. |
| FR-API-002 | The API shall support triggering ingest/process jobs and querying videos, trajectories, analytics, events, and insights. |
| FR-API-003 | The API shall accept a user video upload, return a job identifier immediately, process the video asynchronously, and expose job status so no client request blocks on inference. |

---

## 8. Dashboard & Reporting

| ID | Requirement |
|----|-------------|
| FR-UI-001 | Overview page shall display total vehicles, traffic state, scene context, and major events. |
| FR-UI-002 | Traffic Flow page shall display volume, speed trends, congestion timeline, and flow–density views. |
| FR-UI-003 | Bottleneck page shall display heatmaps, congestion locations, and accumulation. |
| FR-UI-004 | Flow Imbalance page shall display lane utilization (and direction distribution when available). |
| FR-UI-005 | Events page shall display detected events, timestamps, zone/location, and severity. |
| FR-UI-006 | Reports page shall display automated summaries and key findings. |
| FR-UI-007 | Users shall be able to upload their own video, or select a previously processed one, and trigger Traffilytics processing. |
| FR-UI-008 | The dashboard shall show job progress/status for an in-flight upload rather than a blocking wait. |
| FR-RPT-001 | The system shall generate automated reports/summaries for transportation analysis. |

---

## 9. Training, Evaluation & Deployment

| ID | Requirement |
|----|-------------|
| FR-ML-001 | Training pipelines shall use PyTorch / YOLO training tooling on the UAV-OBB annotation splits. |
| FR-ML-002 | Evaluation pipelines shall report detection metrics on the held-out UAV-OBB `valid` split and tracking diagnostics on video. |
| FR-DEP-001 | The application shall be structured as modular, deployable services (e.g., Dockerized). |

---

## 10. End-to-End MVP

| ID | Requirement |
|----|-------------|
| FR-MVP-001 | A complete run from user-uploaded video → Traffilytics detect/track/trajectories → analytics → DB/API → dashboard/reports shall be achievable for at least one clip. |
