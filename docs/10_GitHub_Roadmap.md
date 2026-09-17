# GitHub Roadmap

## Suggested Repository Structure

```
traffilytics/
├── data/
│   ├── raw/                  # uploaded video + UAV-OBB demo clips
│   ├── annotations/          # UAV-OBB train, valid, test, data.yaml
│   └── processed/
├── adapters/
│   └── uav_obb/              # download/install, split discovery, data.yaml handling
├── computer_vision/
│   ├── preprocessing/        # Traffilytics ingest, frames, metadata
│   ├── detection/            # Train + infer YOLO OBB
│   ├── tracking/             # ByteTrack (+ optional alternatives)
│   └── trajectories/         # Generation, lane assignment, diagnostics
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
│   └── services/             # jobs / workers
├── frontend/
│   ├── dashboard/
│   └── components/
├── configs/
│   └── lanes/                # per-video lane/zone polygons
├── models/                   # configs + trained weights
├── tests/
├── docker/                   # Dockerfiles / compose
└── docs/
```

UAV-OBB is an **external dataset** installed into `data/annotations/` (splits) and `data/raw/` (bundled demo clips) by `scripts/download_uav_obb.py`; it is not committed to the repository.

---

## Epics & Issues

### Epic 1 — Dataset & Video Pipeline

| Issue | Description |
|-------|-------------|
| UAV-OBB access & layout | `scripts/download_uav_obb.py` (Mendeley, checksum-verified) → `train/valid/test` + `data.yaml` |
| Modular video ingestion | Traffilytics OpenCV-based load, frames, metadata |
| Artifact management | raw / datasets / processed separation |
| Input policy | Accept video as supplied; no stabilization R&D |

**Deliverables:** Clips loadable through Traffilytics preprocessing with stored metadata.

---

### Epic 2 — Detection Training & Evaluation

| Issue | Description |
|-------|-------------|
| Annotation adapter | UAV-OBB YOLO-OBB labels + `data.yaml` → training dataset |
| Six-class map | Align `CLASS_NAMES` with `0 bike … 5 truck` |
| Train YOLO OBB | Train **your** model on UAV-OBB splits (PyTorch/Ultralytics) |
| Evaluate detector | mAP50, mAP50-95, precision, recall, per-class + overlays |
| Optional baseline | Compare to a public pretrained OBB checkpoint without adopting it as the product model |

**Deliverables:** Trained weights in `models/` + evaluation report.

---

### Epic 3 — Tracking & Trajectory Generation

| Issue | Description |
|-------|-------------|
| Integrate ByteTrack | Independent integration into Traffilytics pipeline |
| Optional tracker comparison | OC-SORT / DeepSORT evaluation |
| Generate trajectories | From your detections + tracks |
| Lane polygons + assigner | Per-video lane/zone config and point-in-polygon assignment |
| Tracking diagnostics | Track counts, ID switches, fragmentation, overlay video on UAV-OBB clips |

**Deliverables:** Generated trajectories + tracking stability notes.

---

### Epic 4 — Analytics Engine

| Issue | Description |
|-------|-------------|
| Flow characterization | Own volume/speed/density/state + flow–density |
| Scale & units | Optional pixel-to-metre scale; label pixel-based outputs |
| Bottleneck detection | Own zone methodology |
| Flow imbalance module | Dedicated lane imbalance feature over configured lanes |
| Event detection | Rule-based stopped / sudden congestion / spillback |
| Insights | Template-based automated summaries |

**Deliverables:** Analytics + events + insights from **generated** trajectories.

---

### Epic 5 — Backend, Database & Dashboard

| Issue | Description |
|-------|-------------|
| Database schema | Videos, vehicles, trajectories, analytics, events, eval runs |
| Upload + job queue | Accept uploads, enqueue work, report progress; no blocking requests |
| FastAPI services | Jobs, query APIs, process triggers |
| Dashboard | Upload/Jobs, Overview, Flow, Bottleneck, Imbalance, Events, Reports |
| Lane editor or config loader | Supply lane polygons per video |
| Automated reports | Transportation analysis summaries |

**Deliverables:** End-to-end API + UI on stored platform outputs.

---

### Epic 6 — Packaging, Testing & Hardening

| Issue | Description |
|-------|-------------|
| Dockerize services | API, worker, DB, frontend as modular deployables |
| Detection/tracking tests | CPU-only harnesses; no dataset download in CI |
| Analytics tests | Scenario checks |
| Performance baselines | Runtime per clip; train notes |
| Licence compliance | Document Ultralytics AGPL-3.0 obligations and UAV-OBB attribution |

**Deliverables:** Deployable compose stack + test/benchmark docs.

---

## Suggested Sequencing

```
Epic 1 (Video pipeline + UAV-OBB layout)
  → Epic 2 (Train/eval OBB)
    → Epic 3 (Track + generate trajectories + lanes + diagnostics)
      → Epic 4 (Analytics + insights)
        → Epic 5 (FastAPI + DB + upload/jobs + Dashboard + reports)
          → Epic 6 (Docker + tests)
```

Detector training is an offline GPU step and can run on a free hosted GPU; the rest of the roadmap is CPU-friendly.

---

## MVP Checklist

- [ ] Traffilytics video ingestion/preprocessing works on uploaded footage
- [ ] YOLO OBB model trained on UAV-OBB annotations and evaluated
- [ ] Tracker integrated; trajectories **generated** by the platform
- [ ] Tracking stability reviewed on UAV-OBB video clips
- [ ] Lane/zone polygons configurable per video
- [ ] Custom flow, bottleneck, imbalance, and event analytics implemented
- [ ] Automated insights generated
- [ ] Upload → job → results flow works without blocking requests
- [ ] Results in DB, exposed via FastAPI, visible on dashboard
- [ ] Reports available; app layout Docker-ready

---

## Issue Labels

| Label | Use |
|-------|-----|
| `epic-1` … `epic-6` | Epic membership |
| `dataset` | UAV-OBB access / annotation adapters |
| `cv-train` / `cv-track` | Detection training / tracking |
| `analytics` / `backend` / `frontend` / `devops` | Area |
| `mvp` | Required for MVP |
| `evaluation` | Detector metrics and tracking diagnostics |

---

## Attribution

UAV-OBB is CC BY 4.0 — cite Ahmad, Fengjun, Bibi & Slaman Pathan (2026), Mendeley Data V3, [doi:10.17632/6snrjwcpkh.3](https://doi.org/10.17632/6snrjwcpkh.3), and state any modifications. Ultralytics is AGPL-3.0; the project licence must be compatible.
