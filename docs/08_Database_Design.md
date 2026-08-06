# Database Design

## Purpose

Traffilytics maintains its **own persistent analytics database** for videos, **generated** trajectories, events, and analytics. DRIFT does not provide this database. DRIFT ground-truth trajectory CSVs are used for **evaluation/benchmarking**, not as the live trajectory store.

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
| `site` | DRIFT site id (when source is DRIFT) |
| `location` | Human-readable label |
| `duration` | Seconds |
| `fps` | e.g., 30 |
| `resolution` | e.g., `3840x2160` |
| `source` | e.g., `drift` |
| `stabilized` | Whether input was pre-stabilized |
| `status` | `queued` / `processing` / `completed` / `failed` |
| `model_version` | Detector weights id used for processing |
| `tracker_name` | e.g., `bytetrack` |

---

### Vehicles

Tracks produced by Traffilytics (not imported GT as source of truth).

| Column | Description |
|--------|-------------|
| `video_id` | FK |
| `track_id` | Platform track id |
| `class_id` | `0` bus, `1` car, `2` truck |
| `vehicle_type` | Label |
| `entry_frame` / `exit_frame` | Span |
| `entry_time` / `exit_time` | Optional |

**Primary key:** `(video_id, track_id)`

---

### Trajectories

Generated pose stream from detector + tracker. Pixel space unless a later geo layer is added.

| Column | Description |
|--------|-------------|
| `video_id`, `track_id`, `frame` | Identity |
| `center_x`, `center_y` | Center |
| `width`, `height`, `angle` | OBB parameters |
| `x1,y1` … `x4,y4` | Corners (optional storage) |
| `confidence`, `class_id` | Detection attrs |
| `site`, `lane` | Context / inferred or assigned lane |
| `speed`, `acceleration`, `heading` | Optional derived motion |

---

### Analytics

| Column | Description |
|--------|-------------|
| `video_id` | FK |
| `site` | Optional |
| `frame_start` / `frame_end` | Window |
| `vehicle_count` | Volume |
| `average_speed` | Mean speed |
| `density` | Density metric |
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
| `site`, `location` | Where |
| `severity` | Severity |
| `track_id` | Optional related track |

---

### EvaluationRuns (optional but recommended)

Stores benchmarks against DRIFT GT—**separate** from live trajectory tables.

| Column | Description |
|--------|-------------|
| `eval_id` | PK |
| `video_id` | FK |
| `model_version` | Detector id |
| `tracker_name` | Tracker id |
| `detection_metrics` | JSON (e.g., mAP) |
| `tracking_metrics` | JSON (e.g., MOTA/IDF1 or custom) |
| `trajectory_metrics` | JSON vs GT CSV |
| `created_at` | Timestamp |

---

## Example Logical Schema (SQL-ish)

```sql
CREATE TABLE videos (
  video_id      TEXT PRIMARY KEY,
  site          TEXT,
  location      TEXT,
  duration      REAL,
  fps           REAL,
  resolution    TEXT,
  source        TEXT,
  stabilized    BOOLEAN,
  status        TEXT,
  model_version TEXT,
  tracker_name  TEXT
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
  site         TEXT,
  lane         INTEGER,
  speed        REAL,
  acceleration REAL,
  heading      REAL,
  PRIMARY KEY (video_id, track_id, frame),
  FOREIGN KEY (video_id, track_id)
    REFERENCES vehicles(video_id, track_id)
);

CREATE TABLE analytics (
  video_id         TEXT REFERENCES videos(video_id),
  site             TEXT,
  frame_start      INTEGER,
  frame_end        INTEGER,
  vehicle_count    INTEGER,
  average_speed    REAL,
  density          REAL,
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
  site       TEXT,
  location   TEXT,
  severity   TEXT,
  track_id   INTEGER
);

CREATE TABLE evaluation_runs (
  eval_id             TEXT PRIMARY KEY,
  video_id            TEXT REFERENCES videos(video_id),
  model_version       TEXT,
  tracker_name        TEXT,
  detection_metrics   JSON,
  tracking_metrics    JSON,
  trajectory_metrics  JSON,
  created_at          TEXT
);
```

---

## Indexing Guidance

- `(video_id, frame)` on trajectories for window queries
- `(video_id, lane)` for imbalance aggregations
- `(video_id)` on analytics and events
- `(model_version)` on videos / evaluation_runs for experiment traceability

## Data Lifecycle

1. Register video metadata on ingest
2. After Traffilytics detect+track, write vehicles + trajectories
3. Optionally run GT benchmark → `evaluation_runs`
4. Analytics engine writes analytics + events
5. Insights/reports read aggregates via API
6. DRIFT GT CSVs remain files (or staging tables) for eval—not the dashboard trajectory source
