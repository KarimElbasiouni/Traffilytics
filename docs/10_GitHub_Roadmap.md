# GitHub Roadmap

## Suggested Repository Structure

```
traffilytics/
├── data/
│   ├── raw/                  # DRIFT videos
│   ├── annotations/          # DRIFT OBB labels / splits
│   ├── gt_trajectories/      # DRIFT GT CSVs (eval only)
│   └── processed/
├── computer_vision/
│   ├── preprocessing/        # Traffilytics ingest, frames, metadata
│   ├── detection/            # Train + infer YOLO OBB
│   ├── tracking/             # ByteTrack (+ optional alternatives)
│   └── trajectories/         # Generation + GT benchmarking
├── analytics/
│   ├── traffic_flow/
│   ├── bottleneck/
│   ├── imbalance/
│   ├── events/
│   ├── micro/                # optional LC/TTC
│   └── insights/
├── backend/
│   ├── api/                  # FastAPI
│   ├── database/
│   └── services/
├── frontend/
│   ├── dashboard/
│   └── components/
├── models/                   # configs + trained weights
├── tests/
├── docker/                   # Dockerfiles / compose
└── docs/
```

DRIFT upstream ([AIxMobility/The-DRIFT](https://github.com/AIxMobility/The-DRIFT)) is an **external dataset/reference**, not this app’s root.

---

## Epics & Issues

### Epic 1 — Dataset & Video Pipeline

| Issue | Description |
|-------|-------------|
| DRIFT access & layout | HF/local paths for videos, annotations, GT CSVs |
| Modular video ingestion | Traffilytics OpenCV-based load, frames, metadata |
| Artifact management | raw / annotations / gt_trajectories / processed |
| Stabilized input policy | Prefer DRIFT stabilized videos; no Stabilo R&D focus |

**Deliverables:** Clips loadable through Traffilytics preprocessing with stored metadata.

---

### Epic 2 — Detection Training & Evaluation

| Issue | Description |
|-------|-------------|
| Annotation adapter | DRIFT OBB format → training dataset |
| Train YOLO OBB | Train **your** model on DRIFT splits (PyTorch/YOLO tooling) |
| Evaluate detector | Metrics + overlays vs held-out annotations |
| Optional baseline | Compare to DRIFT `best.pt` without adopting it as the product model |

**Deliverables:** Trained weights in `models/` + evaluation report.

---

### Epic 3 — Tracking & Trajectory Generation

| Issue | Description |
|-------|-------------|
| Integrate ByteTrack | Independent integration into Traffilytics pipeline |
| Optional tracker comparison | OC-SORT / DeepSORT evaluation |
| Generate trajectories | From your detections + tracks |
| Benchmark vs GT | Compare to DRIFT trajectory CSVs (validation only) |
| Lane utilization hooks | Infer or attach lane info for analytics/viz |

**Deliverables:** Generated trajectories + benchmark notes.

---

### Epic 4 — Analytics Engine

| Issue | Description |
|-------|-------------|
| Flow characterization | Own volume/speed/density/state + flow–density |
| Bottleneck detection | Own zone methodology |
| Flow imbalance module | Dedicated lane imbalance feature |
| Event detection | Rule-based stopped / sudden congestion / spillback |
| Insights | Template-based automated summaries |

**Deliverables:** Analytics + events + insights from **generated** trajectories.

---

### Epic 5 — Backend, Database & Dashboard

| Issue | Description |
|-------|-------------|
| Database schema | Videos, vehicles, trajectories, analytics, events, eval runs |
| FastAPI services | Jobs, query APIs, process triggers |
| Dashboard | Overview, Flow, Bottleneck, Imbalance, Events, Reports |
| Automated reports | Transportation analysis summaries |

**Deliverables:** End-to-end API + UI on stored platform outputs.

---

### Epic 6 — Packaging, Testing & Hardening

| Issue | Description |
|-------|-------------|
| Dockerize services | API, worker, DB, frontend as modular deployables |
| Detection/tracking tests | Eval harnesses against DRIFT |
| Analytics tests | Scenario checks |
| Performance baselines | Runtime per clip; train notes |

**Deliverables:** Deployable compose stack + test/benchmark docs.

---

## Suggested Sequencing

```
Epic 1 (Video pipeline + DRIFT data layout)
  → Epic 2 (Train/eval OBB)
    → Epic 3 (Track + generate trajectories + GT benchmark)
      → Epic 4 (Analytics + insights)
        → Epic 5 (FastAPI + DB + Dashboard + reports)
          → Epic 6 (Docker + tests)
```

Do **not** sequence the product around “CSV ingest as the live path.” GT CSVs enter at Epic 3 as **benchmark inputs** only.

---

## MVP Checklist

- [ ] Traffilytics video ingestion/preprocessing works on DRIFT footage
- [ ] YOLO OBB model trained on DRIFT annotations and evaluated
- [ ] Tracker integrated; trajectories **generated** by the platform
- [ ] Generated trajectories benchmarked against DRIFT GT (as available)
- [ ] Custom flow, bottleneck, imbalance, and event analytics implemented
- [ ] Automated insights generated
- [ ] Results in DB, exposed via FastAPI, visible on dashboard
- [ ] Reports available; app layout Docker-ready

---

## Issue Labels

| Label | Use |
|-------|-----|
| `epic-1` … `epic-6` | Epic membership |
| `drift-data` | Dataset/annotation/GT access |
| `cv-train` / `cv-track` | Detection training / tracking |
| `analytics` / `backend` / `frontend` / `devops` | Area |
| `mvp` | Required for MVP |
| `benchmark` | GT comparison work |

---

## Attribution

Cite DRIFT ([arXiv:2504.11019](https://arxiv.org/abs/2504.11019)) when publishing. Acknowledge Stabilo only if its stabilization code is used.
