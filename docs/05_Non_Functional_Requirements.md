# Non-Functional Requirements

Non-functional requirements for Traffilytics as a **complete, deployable traffic intelligence platform** that uses DRIFT for training and evaluation.

---

## 1. Performance

| ID | Requirement |
|----|-------------|
| NFR-PERF-001 | The pipeline shall process offline video batches; real-time emergency latency is out of scope. |
| NFR-PERF-002 | Frame extraction and detection shall support DRIFT 4K sources with configurable downscaling for throughput. |
| NFR-PERF-003 | Tracking and analytics shall run on **generated** trajectories without interactive frame-by-frame user input. |
| NFR-PERF-004 | Dashboard/API queries for summary metrics should return within interactive bounds for MVP clip sizes. |
| NFR-PERF-005 | Training may be GPU-bound and offline; inference jobs should report progress/status via the backend. |

---

## 2. Scalability & Capacity

| ID | Requirement |
|----|-------------|
| NFR-SCALE-001 | Architecture shall support multiple DRIFT sites/clips as independent jobs. |
| NFR-SCALE-002 | Storage shall handle high-volume generated trajectory time series (30 fps × many tracks). |
| NFR-SCALE-003 | Services (API, worker/pipeline, DB, frontend) shall be separable for horizontal growth later. |

---

## 3. Accuracy & Reliability

| ID | Requirement |
|----|-------------|
| NFR-ACC-001 | Detection confidence thresholds shall be configurable; training/eval metrics shall be recorded. |
| NFR-ACC-002 | Tracking shall preserve `track_id` sufficiently for analytics; tracker choice shall be evaluable. |
| NFR-ACC-003 | Analytics outputs shall be reproducible for the same video, model weights, and configuration. |
| NFR-ACC-004 | Failures (corrupt video, training error, model load error) shall fail gracefully with clear status/errors. |
| NFR-ACC-005 | Generated trajectories shall be benchmarkable against DRIFT GT CSVs; GT shall not silently replace live outputs. |

---

## 4. Usability

| ID | Requirement |
|----|-------------|
| NFR-USE-001 | Dashboard shall present Overview, Flow, Bottleneck, Imbalance, Events, and Reports. |
| NFR-USE-002 | Insights and reports shall be readable by non-CV specialists. |
| NFR-USE-003 | Job status (ingest, train optional, process, complete/fail) shall be visible to technical users. |

---

## 5. Maintainability & Modularity

| ID | Requirement |
|----|-------------|
| NFR-MAINT-001 | Video processing, detection, tracking, analytics, backend, and frontend shall be separable modules. |
| NFR-MAINT-002 | Detector weights and tracker implementations shall be swappable behind clear interfaces. |
| NFR-MAINT-003 | Insight generation shall start rule-based, with an extension point for LLM integration. |
| NFR-MAINT-004 | DRIFT-specific dataset adapters (paths, annotation loaders, GT benchmark loaders) shall be isolated from core platform logic. |
| NFR-MAINT-005 | Dependencies on DRIFT research scripts shall be minimized; prefer Traffilytics-owned implementations. |

---

## 6. Portability & Environment

| ID | Requirement |
|----|-------------|
| NFR-PORT-001 | Training and inference shall run with PyTorch; GPU optional but recommended for YOLO OBB training. |
| NFR-PORT-002 | Video I/O shall use standard libraries (e.g., OpenCV) for MP4 and frame extraction. |
| NFR-PORT-003 | Configuration (dataset paths, thresholds, zones, model paths) shall be externalized. |
| NFR-PORT-004 | Services shall be packageable with Docker (or equivalent) for reproducible deployment. |

---

## 7. Security & Privacy (PoC Level)

| ID | Requirement |
|----|-------------|
| NFR-SEC-001 | MVP uses open DRIFT data under its published terms; cite DRIFT when publishing results. |
| NFR-SEC-002 | Local/demo API may be unauthenticated; production auth is future work. |
| NFR-SEC-003 | Secrets (HF tokens, optional LLM keys) shall not be committed to source control. |

---

## 8. Testability & Evaluation

| ID | Requirement |
|----|-------------|
| NFR-TEST-001 | Detector evaluation shall use DRIFT annotation splits / overlays. |
| NFR-TEST-002 | Trajectory/tracking benchmarks shall compare generated tracks to DRIFT GT CSVs. |
| NFR-TEST-003 | Analytics shall be testable with known scenarios or synthetic trajectory subsets. |
| NFR-TEST-004 | Performance benchmarks (runtime per video minute; train time) shall be measurable. |

---

## 9. Documentation & Attribution

| ID | Requirement |
|----|-------------|
| NFR-DOC-001 | Architecture, APIs, schema, and roadmap shall live under `docs/`. |
| NFR-DOC-002 | Docs shall clearly separate DRIFT (dataset/reference) from Traffilytics (platform). |
| NFR-DOC-003 | Acknowledge DRIFT authors; acknowledge Stabilo only if its code/process is used for stabilization. |
