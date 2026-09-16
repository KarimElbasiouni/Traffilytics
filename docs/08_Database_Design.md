# Database Design

## Purpose

Traffilytics maintains its **own persistent analytics database** for videos, **generated** trajectories, events, and analytics. Every trajectory row is produced by the platform's detector and tracker; no external trajectory dataset is imported.

## Entity Overview

```
Videos 1──* Vehicles (generated tracks) 1──* Trajectories (generated)
Videos 1──* Analytics
Videos 1──* Events
Videos 1──* EvaluationRuns (optional)
```

---

## Tables

### Videos

| Column | Description |
|--------|-------------|
| `video_id` | Primary key / clip identifier |
| `site` | Optional user-supplied scene label |
| `location` | Human-readable label |
| `duration` | Seconds |
| `fps` | e.g., 30 |
| `resolution` | e.g., `1920x1080` |
| `source` | e.g., `upload`, `uav_obb_sample` |
| `stabilized` | Whether input was pre-stabilized |
| `status` | `queued` / `processing` / `completed` / `failed` |
| `scale_m_per_px` | Optional pixel-to-metre scale; `NULL` means pixel units |
| `lane_config` | JSON lane/zone polygons for this scene (or a config reference) |
| `model_version` | Detector weights id used for processing |
| `tracker_name` | e.g., `bytetrack` |

`status` is what the dashboard polls while an upload is being processed (FR-API-003).

---

### Vehicles

Tracks produced by Traffilytics.

| Column | Description |
|--------|-------------|
| `video_id` | FK |
| `track_id` | Platform track id |
| `class_id` | `0` bike, `1` bus, `2` car, `3` other_vehicle, `4` taxi, `5` truck |
| `vehicle_type` | Label |
| `entry_frame` / `exit_frame` | Span |
| `entry_time` / `exit_time` | Optional |

**Primary key:** `(video_id, track_id)`

---

### Trajectories

Generated pose stream from detector + tracker. Pixel space unless `scale_m_per_px` is set on the video.

| Column | Description |
|--------|-------------|
| `video_id`, `track_id`, `frame` | Identity |
| `center_x`, `center_y` | Center |
| `width`, `height`, `angle` | OBB parameters |
| `x1,y1` … `x4,y4` | Corners (optional storage) |
| `confidence`, `class_id` | Detection attrs |
| `lane` | Lane label resolved from the video's configured polygons |
| `zone_id` | Optional analysis zone |
| `speed`, `acceleration`, `heading` | Optional derived motion (pixel units unless scaled) |

---

### Analytics

| Column | Description |
|--------|-------------|
| `video_id` | FK |
| `frame_start` / `frame_end` | Window |
| `vehicle_count` | Volume |
| `average_speed` | Mean speed |
| `density` | Density metric |
| `units` | `px` or `m` — how `average_speed` / `density` are expressed |
| `traffic_state` | Congestion state label |
| `congestion_score` | Numeric score |
| `zone_id` | Optional zone |

---

### Events

| Column | Description |
|--------|-------------|
| `event_id` | PK |
| `video_id` | FK |
| `event_type` | `sudden_congestion`, `stopped_vehicle`, `queue_spillback`, … |
| `frame` / `timestamp` | When |
| `zone_id`, `location` | Where |
| `severity` | Severity |
| `track_id` | Optional related track |

---

### EvaluationRuns (optional but recommended)

Model quality records, kept **separate** from live trajectory tables.

| Column | Description |
|--------|-------------|
| `eval_id` | PK |
| `video_id` | FK (nullable — detector eval is on images, not video) |
| `model_version` | Detector id |
| `tracker_name` | Tracker id |
| `detection_metrics` | JSON — mAP50, mAP50-95, precision, recall, per-class |
| `tracking_diagnostics` | JSON — track counts, ID switches, fragmentation |
| `created_at` | Timestamp |

---

## Example Logical Schema (SQL-ish)

```sql
CREATE TABLE videos (
  video_id       TEXT PRIMARY KEY,
  site           TEXT,
  location       TEXT,
  duration       REAL,
  fps            REAL,
  resolution     TEXT,
  source         TEXT,
  stabilized     BOOLEAN,
  status         TEXT,
  scale_m_per_px REAL,
  lane_config    JSON,
  model_version  TEXT,
  tracker_name   TEXT
);

CREATE TABLE vehicles (
  video_id     TEXT REFERENCES videos(video_id),
  track_id     INTEGER,
  class_id     INTEGER,
  vehicle_type TEXT,
  entry_frame  INTEGER,
  exit_frame   INTEGER,
  entry_time   TEXT,
  exit_time    TEXT,
  PRIMARY KEY (video_id, track_id)
);

CREATE TABLE trajectories (
  video_id     TEXT,
  track_id     INTEGER,
  frame        INTEGER,
  center_x     REAL,
  center_y     REAL,
  width        REAL,
  height       REAL,
  angle        REAL,
  confidence   REAL,
  class_id     INTEGER,
  lane         TEXT,
  zone_id      TEXT,
  speed        REAL,
  acceleration REAL,
  heading      REAL,
  PRIMARY KEY (video_id, track_id, frame),
  FOREIGN KEY (video_id, track_id)
    REFERENCES vehicles(video_id, track_id)
);

CREATE TABLE analytics (
  video_id         TEXT REFERENCES videos(video_id),
  frame_start      INTEGER,
  frame_end        INTEGER,
  vehicle_count    INTEGER,
  average_speed    REAL,
  density          REAL,
  units            TEXT,
  traffic_state    TEXT,
  congestion_score REAL,
  zone_id          TEXT
);

CREATE TABLE events (
  event_id   TEXT PRIMARY KEY,
  video_id   TEXT REFERENCES videos(video_id),
  event_type TEXT,
  frame      INTEGER,
  timestamp  TEXT,
  zone_id    TEXT,
  location   TEXT,
  severity   TEXT,
  track_id   INTEGER
);

CREATE TABLE evaluation_runs (
  eval_id              TEXT PRIMARY KEY,
  video_id             TEXT REFERENCES videos(video_id),
  model_version        TEXT,
  tracker_name         TEXT,
  detection_metrics    JSON,
  tracking_diagnostics JSON,
  created_at           TEXT
);
```

---

## Indexing Guidance

- `(video_id, frame)` on trajectories for window queries
- `(video_id, lane)` for imbalance aggregations
- `(video_id)` on analytics and events
- `(model_version)` on videos / evaluation_runs for experiment traceability

## Data Lifecycle

1. Register video metadata on upload; status `queued`
2. Worker sets `processing`, runs Traffilytics detect+track, writes vehicles + trajectories
3. Analytics engine writes analytics + events; status `completed`
4. Optionally record detector metrics / tracking diagnostics → `evaluation_runs`
5. Insights/reports read aggregates via API
6. Detector training metrics stay in `models/runs/` artifacts; only summaries are mirrored into `evaluation_runs`
