# API Design

## Purpose

Traffilytics exposes a backend API (e.g., **FastAPI**) for processing, storage, and data access. DRIFT does not provide this API. Endpoints below may be refined during backend implementation.

## Conventions

| Item | Convention |
|------|------------|
| Style | REST/JSON |
| Base path | `/api/v1` |
| IDs | `video_id`, `track_id`, `event_id`, `site`, `model_version` |
| Classes | `class_id`: `0` bus, `1` car, `2` truck |
| Trajectories | Always **generated** unless an endpoint is explicitly under `/evaluation` |
| Errors | `{ "error": { "code": "...", "message": "..." } }` |

---

## 1. Videos & Jobs

### `POST /api/v1/videos`

Register / upload a traffic video for processing.

**Response `201`**

```json
{
  "video_id": "site_03_clip_01",
  "site": "03",
  "fps": 30,
  "resolution": "3840x2160",
  "duration": 900,
  "source": "drift",
  "status": "queued"
}
```

### `GET /api/v1/videos`

List clips; filter `?site=`

### `GET /api/v1/videos/{video_id}`

Metadata, `model_version`, `tracker_name`, status.

### `POST /api/v1/videos/{video_id}/process`

Run Traffilytics pipeline: detect (trained OBB) → track → generate trajectories → analytics → insights.

**Body (example)**

```json
{
  "model_version": "obb_v1",
  "tracker": "bytetrack"
}
```

### `GET /api/v1/sites`

List known sites.

---

## 2. Training & Evaluation (platform ML)

### `POST /api/v1/models/train` (optional for MVP UI; required as CLI/job)

Start / register OBB training on DRIFT annotation splits.

### `GET /api/v1/models`

List trained model versions available for inference.

### `POST /api/v1/videos/{video_id}/evaluate`

Benchmark current (or specified) model/tracker outputs against DRIFT GT annotations/trajectories.

```json
{
  "eval_id": "eval_001",
  "detection_metrics": {},
  "tracking_metrics": {},
  "trajectory_metrics": {}
}
```

### `GET /api/v1/videos/{video_id}/evaluations`

List evaluation runs for a clip.

---

## 3. Detections, Tracks & Generated Trajectories

### `GET /api/v1/videos/{video_id}/detections`

Paginated OBB detections from Traffilytics model.

### `GET /api/v1/videos/{video_id}/vehicles`

Generated tracks.

### `GET /api/v1/videos/{video_id}/vehicles/{track_id}/trajectory`

Generated trajectory points (`?stride=` optional).

```json
{
  "track_id": 52,
  "class_id": 1,
  "points": [
    {
      "frame": 1204,
      "center_x": 145,
      "center_y": 320,
      "lane": 2,
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
| Overview | `/overview`, `/events` |
| Traffic Flow | `/analytics/flow`, `/analytics/flow-density` |
| Bottleneck | `/analytics/bottlenecks`, `/analytics/heatmap` |
| Flow Imbalance | `/analytics/imbalance` |
| Events | `/events` |
| Reports | `/insights`, `/reports` |
| Eval (internal) | `/evaluate`, `/evaluations` |

---

## 7. Internal Service Boundaries

| Service | Role |
|---------|------|
| Video service | Ingest, metadata, status |
| ML service | Train OBB, register weights, evaluate vs DRIFT |
| Pipeline service | Detect → track → generate trajectories → analytics |
| Analytics service | Flow, bottleneck, imbalance, events |
| Insight / report service | Summaries and report payloads |
| Persistence | Database access layer |

See [07_Component_Design.md](./07_Component_Design.md).
