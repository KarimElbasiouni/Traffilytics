# API Design

## Purpose

Traffilytics exposes a backend API (e.g., **FastAPI**) for uploads, processing, storage, and data access. Endpoints below may be refined during backend implementation.

## Conventions

| Item | Convention |
|------|------------|
| Style | REST/JSON |
| Base path | `/api/v1` |
| IDs | `video_id`, `job_id`, `track_id`, `event_id`, `model_version` |
| Classes | `class_id`: `0` bike, `1` bus, `2` car, `3` other_vehicle, `4` taxi, `5` truck |
| Trajectories | Always **generated** by the platform |
| Units | Analytics payloads carry a `units` field (`px` or `m`) |
| Long work | Never inline; upload and process return a job to poll |
| Errors | `{ "error": { "code": "...", "message": "..." } }` |

---

## 1. Videos & Jobs

### `POST /api/v1/videos`

Upload a traffic video for processing. Accepts `multipart/form-data`. Returns immediately; decoding and inference happen in a worker.

**Response `202`**

```json
{
  "video_id": "uav_clip_01",
  "job_id": "job_7f21",
  "site": "chongqing_arterial",
  "fps": 30,
  "resolution": "1920x1080",
  "duration": 45,
  "source": "upload",
  "status": "queued"
}
```

### `GET /api/v1/videos`

List clips.

### `GET /api/v1/videos/{video_id}`

Metadata, `model_version`, `tracker_name`, `scale_m_per_px`, status.

### `POST /api/v1/videos/{video_id}/process`

Enqueue the Traffilytics pipeline: detect (trained OBB) → track → generate trajectories → analytics → insights. Returns a `job_id`.

**Body (example)**

```json
{
  "model_version": "obb_v1",
  "tracker": "bytetrack",
  "scale_m_per_px": 0.062,
  "lane_config": "configs/lanes/uav_clip_01.json"
}
```

### `GET /api/v1/jobs/{job_id}`

Job status for an in-flight upload or process run.

```json
{
  "job_id": "job_7f21",
  "video_id": "uav_clip_01",
  "stage": "tracking",
  "progress": 0.42,
  "status": "processing"
}
```

### `PUT /api/v1/videos/{video_id}/lanes`

Store or replace the lane/zone polygons for a scene.

---

## 2. Training & Evaluation (platform ML)

### `POST /api/v1/models/train` (optional for MVP UI; required as CLI/job)

Start / register OBB training on the UAV-OBB annotation splits.

### `GET /api/v1/models`

List trained model versions available for inference.

### `POST /api/v1/models/{model_version}/evaluate`

Evaluate a model on held-out UAV-OBB labels.

```json
{
  "eval_id": "eval_001",
  "detection_metrics": {
    "mAP50": 0.0,
    "mAP50-95": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "per_class": {}
  }
}
```

### `POST /api/v1/videos/{video_id}/tracking-diagnostics`

Report ID-stability diagnostics for a processed clip (track counts, ID switches, fragmentation). No trajectory ground truth is assumed.

### `GET /api/v1/videos/{video_id}/evaluations`

List evaluation runs associated with a clip.

---

## 3. Detections, Tracks & Generated Trajectories

### `GET /api/v1/videos/{video_id}/detections`

Paginated OBB detections from the Traffilytics model.

### `GET /api/v1/videos/{video_id}/vehicles`

Generated tracks.

### `GET /api/v1/videos/{video_id}/vehicles/{track_id}/trajectory`

Generated trajectory points (`?stride=` optional).

```json
{
  "track_id": 52,
  "class_id": 2,
  "points": [
    {
      "frame": 1204,
      "center_x": 145,
      "center_y": 320,
      "lane": "lane_2",
      "angle": 0.41,
      "confidence": 0.93
    }
  ]
}
```

---

## 4. Analytics

### `GET /api/v1/videos/{video_id}/analytics/flow`

### `GET /api/v1/videos/{video_id}/analytics/flow-density`

### `GET /api/v1/videos/{video_id}/analytics/bottlenecks`

### `GET /api/v1/videos/{video_id}/analytics/imbalance`

### `GET /api/v1/videos/{video_id}/analytics/heatmap`

### `GET /api/v1/videos/{video_id}/analytics/micro` (optional LC/TTC)

### `GET /api/v1/videos/{video_id}/overview`

Speed and density fields are accompanied by `units`; clients must not present `px` values as physical measurements.

---

## 5. Events, Insights & Reports

### `GET /api/v1/videos/{video_id}/events`

### `GET /api/v1/videos/{video_id}/events/{event_id}`

### `GET /api/v1/videos/{video_id}/insights`

### `GET /api/v1/videos/{video_id}/reports`

Automated transportation analysis report payload.

---

## 6. Dashboard Mapping

| Dashboard page | Primary endpoints |
|----------------|-------------------|
| Upload / Jobs | `POST /videos`, `/jobs/{job_id}` |
| Overview | `/overview`, `/events` |
| Traffic Flow | `/analytics/flow`, `/analytics/flow-density` |
| Bottleneck | `/analytics/bottlenecks`, `/analytics/heatmap` |
| Flow Imbalance | `/analytics/imbalance`, `PUT /videos/{id}/lanes` |
| Events | `/events` |
| Reports | `/insights`, `/reports` |
| Eval (internal) | `/models/{v}/evaluate`, `/evaluations` |

---

## 7. Internal Service Boundaries

| Service | Role |
|---------|------|
| Video service | Upload, metadata, status |
| Job service | Queue, worker dispatch, progress reporting |
| ML service | Train OBB on UAV-OBB, register weights, evaluate |
| Pipeline service | Detect → track → generate trajectories → analytics |
| Analytics service | Flow, bottleneck, imbalance, events |
| Insight / report service | Summaries and report payloads |
| Persistence | Database access layer |

See [07_Component_Design.md](./07_Component_Design.md).
