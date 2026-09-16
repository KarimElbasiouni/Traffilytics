# Project Scope

## Primary Dataset — UAV-OBB (Training & Evaluation)

**UAV-OBB** is the primary dataset for **detector training and evaluation**. At runtime Traffilytics processes **user-uploaded video**; the dataset's bundled MP4 clips serve as demo and sanity-check footage.

| Spec | Value |
|------|-------|
| Source | UAV imagery over urban roads in Chongqing and Wuhan, China |
| Imagery | 1920×1080 JPEG, predominantly nadir view, ~75–108 m altitude |
| Conditions | Morning, midday, evening, night, rain, mist/light fog; wide FOV and zoom (strong scale variation) |
| Annotations | YOLOv8-OBB text labels — class id + four corner vertices in normalized coordinates |
| Volume | ~1.4–1.6k images, ~36–47k oriented instances (varies by release version) |
| Splits | `train/` `valid/` `test/` shipped with the dataset, plus `data.yaml` |
| Classes | bike (`0`), bus (`1`), car (`2`), other_vehicle (`3`), taxi (`4`), truck (`5`) |
| Video | `test_videos_mp4/` — one short clip with sparsely annotated reference frames, plus longer unannotated sequences |
| Trajectory ground truth | **None** |
| Lane annotations | **None** |
| Licence | CC BY 4.0 (attribution required) |
| Download | [Mendeley Data](https://data.mendeley.com/datasets/6snrjwcpkh/3) (DOI 10.17632/6snrjwcpkh) · [Kaggle mirror](https://www.kaggle.com/datasets/mdferozahmedafm/uav-obb-drone-based-urban-vehicle-dataset) |

Image and instance counts differ between the Mendeley release description and the published paper, so the shipped `data.yaml` and split directories are the source of truth — not any figure quoted in these docs.

### Ownership model

| Traffilytics builds | Traffilytics reuses / does not reinvent |
|--------------------|----------------------------------------|
| Video ingestion, preprocessing, frames, metadata | YOLO **architecture** (trains its own OBB weights) |
| YOLO OBB **training & evaluation** on UAV-OBB | ByteTrack (or similar) **algorithm** — integrate & evaluate |
| Tracking integration + **own trajectory generation** | OpenCV, PyTorch, Ultralytics |
| Analytics engine, bottleneck, imbalance, events | UAV-OBB imagery and OBB labels (CC BY 4.0) |
| Lane/zone definition and scale calibration workflow | — |
| Insights, dashboard, FastAPI, database, reports, Docker | — |

---

## In Scope

### Inputs

| Input | Role |
|-------|------|
| User-uploaded aerial/drone video (MP4 or equivalent) | Runtime processing input |
| UAV-OBB `train` / `valid` / `test` OBB splits | Detector training and evaluation |
| UAV-OBB supplementary MP4 clips | Demo footage and tracking sanity checks |
| User-defined lane/zone polygons per video | Lane utilization, imbalance, bottleneck zones |
| Optional pixel-to-metre scale per video | Physical-unit speed and density |

### Processing Pipeline (Traffilytics-owned)

- Modular video ingestion, preprocessing, frame extraction, metadata management
- Accepts video as uploaded; stabilization is not a project focus and is not required
- Train YOLO OBB model on UAV-OBB splits; deploy trained weights for inference
- Independently integrate ByteTrack; optionally compare OC-SORT / DeepSORT
- Generate trajectories from **your** detections + tracks (YOLO-OBB geometry family)
- Resolve lanes and zones from user-supplied polygons
- Custom analytics engine:
  - Traffic flow characterization (own metrics/algorithms)
  - Bottleneck detection (own methodology)
  - Traffic flow imbalance (dedicated feature)
  - Rule-based event detection (stopped vehicle, sudden congestion, queue spillback)
  - Optional micro metrics (LC, TTC) as platform features
- Automated insight generation
- Backend (e.g. FastAPI), persistent database, interactive dashboard, automated reports
- Asynchronous job handling so uploads never block on inference
- Modular deployment packaging (e.g. Dockerized services)

### Outputs

| Output | Description |
|--------|-------------|
| Trained OBB detector | Weights + evaluation metrics on UAV-OBB splits |
| Detection evaluation | mAP50, mAP50-95, precision, recall, per-class breakdown, overlays |
| Tracking diagnostics | Track counts, ID switches, fragmentation, overlay video |
| Generated trajectories | From Traffilytics detector + tracker |
| Traffic statistics | Counts, speeds, density, congestion state |
| Bottleneck & imbalance analysis | Zones/lanes with causes and utilization |
| Event reports | Typed events with time, zone/location, severity |
| Insights & reports | Human-readable summaries |
| Dashboard + API | Interactive exploration of stored results |

### Vehicle Classes (UAV-OBB)

| `class_id` | Label |
|------------|-------|
| 0 | bike |
| 1 | bus |
| 2 | car |
| 3 | other_vehicle |
| 4 | taxi |
| 5 | truck |

---

## Out of Scope

The system will **not**:

- Control traffic lights or signal timing
- Recommend construction or capital projects
- Replace transportation engineers’ judgment
- Predict long-term urban development
- Provide real-time emergency response / dispatch
- Perform autonomous driving functions
- Invent a new detector or tracker architecture
- Make video stabilization a core deliverable
- Produce or curate a ground-truth trajectory dataset
- Report physical-unit speeds or densities for video with no supplied scale

## MVP Boundaries

The MVP is complete when:

1. A user can upload aerial traffic footage and Traffilytics ingests it via its own video pipeline
2. A YOLO OBB model trained on UAV-OBB can detect the six vehicle classes
3. An independently integrated tracker (ByteTrack or evaluated alternative) maintains identities
4. Trajectories are **generated** by Traffilytics from those detections and tracks
5. Custom traffic metrics, bottlenecks, imbalance, and events are computed over user-defined lanes/zones
6. Insights are generated
7. Results are stored, served via backend API, and shown on an interactive dashboard
8. The application is structured for modular deployment (e.g. Docker-ready layout)

## Future Expansion (Not MVP)

- Additional OBB datasets beyond UAV-OBB (behind separate adapters)
- Live / continuous camera streams
- LLM-backed insight generation
- Automatic lane detection instead of manual polygons
- Geo-referenced calibration and map overlays
- Production auth, multi-tenant ops

See also: [11_Future_Work.md](./11_Future_Work.md)
