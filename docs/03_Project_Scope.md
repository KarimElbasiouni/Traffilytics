# Project Scope

## Primary Dataset — DRIFT (Training & Evaluation)

**DRIFT** is the primary dataset for **training**, **evaluation**, and **benchmarking**. Traffilytics does not treat the DRIFT GitHub repo as the application to ship.

| Spec | Value |
|------|-------|
| Sites | 9 interconnected urban intersections, Daejeon, South Korea |
| Imagery | 4K drone footage, ~250 m altitude, frame-level annotations, 30 fps |
| Annotations | Polygon-based oriented bounding boxes (OBB); ~300K annotated instances |
| GT trajectories | ~81,699 tracks — used for **validation/benchmarking only** |
| Classes | Bus (`0`), Car (`1`), Truck (`2`) |
| Downloads | [Hugging Face](https://huggingface.co/datasets/Hj-Lee/The-DRIFT) · [GitHub](https://github.com/AIxMobility/The-DRIFT) |

### Ownership model

| Traffilytics builds | Traffilytics reuses / does not reinvent |
|--------------------|----------------------------------------|
| Video ingestion, preprocessing, frames, metadata | YOLO **architecture** (train own OBB weights) |
| YOLO OBB **training & evaluation** on DRIFT | ByteTrack (or similar) **algorithm** — integrate & evaluate |
| Tracking integration + **own trajectory generation** | OpenCV, PyTorch |
| Analytics engine, bottleneck, imbalance, events | DRIFT **stabilized videos** or their stabilization process (not a focus) |
| Insights, dashboard, FastAPI, database, reports, Docker | DRIFT GT CSVs for **benchmarking only** |

---

## In Scope

### Inputs

| Input | Role |
|-------|------|
| DRIFT drone videos (prefer stabilized where available) | Runtime processing input |
| DRIFT OBB annotations / train–val–test splits | Detector training and evaluation |
| DRIFT ground-truth trajectory CSVs | Validation and benchmarking of **generated** trajectories |
| Site reference imagery / RoI configs (as needed) | Zones, visualization, lane/analytics aids |

### Processing Pipeline (Traffilytics-owned)

- Modular video ingestion, preprocessing, frame extraction, metadata management
- Optional use of already-stabilized DRIFT video (stabilization itself is not a project focus)
- Train YOLO OBB model on DRIFT annotations; deploy trained weights for inference
- Independently integrate ByteTrack; optionally compare OC-SORT / DeepSORT
- Generate trajectories from **your** detections + tracks (same OBB annotation format family as DRIFT)
- Infer or utilize lane information for analytics and visualization
- Custom analytics engine:
  - Traffic flow characterization (own metrics/algorithms)
  - Bottleneck detection (own methodology)
  - Traffic flow imbalance (dedicated feature)
  - Rule-based event detection (stopped vehicle, sudden congestion, queue spillback)
  - Optional micro metrics (LC, TTC) as platform features—not copies of DRIFT notebooks
- Automated insight generation
- Backend (e.g. FastAPI), persistent database, interactive dashboard, automated reports
- Modular deployment packaging (e.g. Dockerized services)

### Outputs

| Output | Description |
|--------|-------------|
| Trained OBB detector | Weights + evaluation metrics on DRIFT splits |
| Tracking evaluation | ID stability / tracking metrics as defined in tests |
| Generated trajectories | From Traffilytics detector + tracker |
| Benchmark reports | Comparison vs DRIFT GT trajectories where applicable |
| Traffic statistics | Counts, speeds, density, congestion state |
| Bottleneck & imbalance analysis | Zones/lanes with causes and utilization |
| Event reports | Typed events with time, location/site, severity |
| Insights & reports | Human-readable summaries |
| Dashboard + API | Interactive exploration of stored results |

### Vehicle Classes (DRIFT)

| `class_id` | Label |
|------------|-------|
| 0 | Bus |
| 1 | Car |
| 2 | Truck |

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
- Use DRIFT provided trajectory CSVs as the live/production trajectory source
- Ship DRIFT’s research notebooks as the product analytics/dashboard layer

## MVP Boundaries

The MVP is complete when:

1. Traffilytics can ingest DRIFT traffic footage via its own video pipeline
2. A YOLO OBB model trained on DRIFT annotations can detect vehicles
3. An independently integrated tracker (ByteTrack or evaluated alternative) maintains identities
4. Trajectories are **generated** by Traffilytics (GT CSVs used for validation)
5. Custom traffic metrics, bottlenecks, imbalance, and events are computed
6. Insights are generated
7. Results are stored, served via backend API, and shown on an interactive dashboard
8. The application is structured for modular deployment (e.g. Docker-ready layout)

## Future Expansion (Not MVP)

- Additional datasets beyond DRIFT
- Live / continuous camera streams
- LLM-backed insight generation
- Deeper micro analytics productization; geo-aligned map overlays
- Production auth, multi-tenant ops

See also: [11_Future_Work.md](./11_Future_Work.md)
