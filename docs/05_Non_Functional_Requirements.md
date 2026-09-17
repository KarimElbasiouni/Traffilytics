# Non-Functional Requirements

Non-functional requirements for Traffilytics as a **complete, deployable traffic intelligence platform** that uses **UAV-OBB** for detector training and evaluation, and processes **user-uploaded video** at runtime.

---

## 1. Performance

| ID | Requirement |
|----|-------------|
| NFR-PERF-001 | The pipeline shall process offline video batches; real-time emergency latency is out of scope. |
| NFR-PERF-002 | Frame extraction and detection shall support HD through 4K sources with configurable downscaling and frame stride for throughput. |
| NFR-PERF-003 | Tracking and analytics shall run on **generated** trajectories without interactive frame-by-frame user input. |
| NFR-PERF-004 | Dashboard/API queries for summary metrics should return within interactive bounds for MVP clip sizes. |
| NFR-PERF-005 | Training may be GPU-bound and offline. Inference shall run as an asynchronous job that reports progress/status via the backend; no HTTP request shall block on video processing. |

---

## 2. Scalability & Capacity

| ID | Requirement |
|----|-------------|
| NFR-SCALE-001 | Architecture shall support multiple videos/clips as independent jobs. |
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
| NFR-ACC-005 | Trajectory and analytics outputs shall record the model weights and configuration that produced them, so every result is attributable to a specific run. |
| NFR-ACC-006 | Where no pixel-to-metre scale is supplied, derived units (speed, density) shall be labelled as pixel-based rather than presented as physical measurements. |
| NFR-ACC-007 | Reported detection metrics shall name the evaluation split and its image count, so small-sample results (such as UAV-OBB's 10-image `test` split) cannot be mistaken for benchmark-grade figures. |

---

## 4. Usability

| ID | Requirement |
|----|-------------|
| NFR-USE-001 | Dashboard shall present Overview, Flow, Bottleneck, Imbalance, Events, and Reports. |
| NFR-USE-002 | Insights and reports shall be readable by non-CV specialists. |
| NFR-USE-003 | Job status (upload, ingest, process, complete/fail) shall be visible to the user who submitted it. |

---

## 5. Maintainability & Modularity

| ID | Requirement |
|----|-------------|
| NFR-MAINT-001 | Video processing, detection, tracking, analytics, backend, and frontend shall be separable modules. |
| NFR-MAINT-002 | Detector weights and tracker implementations shall be swappable behind clear interfaces. |
| NFR-MAINT-003 | Insight generation shall start rule-based, with an extension point for LLM integration. |
| NFR-MAINT-004 | Dataset-specific adapters (paths, annotation loaders, class maps) shall be isolated from core platform logic so another OBB dataset can be added without changing the CV core. |
| NFR-MAINT-005 | Third-party research scripts shall not be vendored as product code; prefer Traffilytics-owned implementations. |

---

## 6. Portability & Environment

| ID | Requirement |
|----|-------------|
| NFR-PORT-001 | Training and inference shall run with PyTorch; GPU optional but recommended for YOLO OBB training. |
| NFR-PORT-002 | Video I/O shall use standard libraries (e.g., OpenCV) for MP4 and frame extraction. |
| NFR-PORT-003 | Configuration (dataset paths, thresholds, lane/zone polygons, scale, model paths) shall be externalized. |
| NFR-PORT-004 | Services shall be packageable with Docker (or equivalent) for reproducible deployment. |

---

## 7. Security & Privacy (PoC Level)

| ID | Requirement |
|----|-------------|
| NFR-SEC-001 | Training data shall be used under its published licence. UAV-OBB is CC BY 4.0: attribute the authors and state any modifications. |
| NFR-SEC-002 | Local/demo API may be unauthenticated; production auth is future work. |
| NFR-SEC-003 | Secrets (API keys, optional LLM keys) shall not be committed to source control. |
| NFR-SEC-004 | User-uploaded video shall be treated as user data: stored only for processing and retrieval by that user, never redistributed or added to training data without consent. |

---

## 8. Testability & Evaluation

| ID | Requirement |
|----|-------------|
| NFR-TEST-001 | Detector evaluation shall use held-out UAV-OBB splits plus qualitative overlays. |
| NFR-TEST-002 | Tracking shall be assessed with ID-stability diagnostics and visual review on video, including UAV-OBB's sparsely annotated reference clip; no trajectory ground-truth dataset is assumed. |
| NFR-TEST-003 | Analytics shall be testable with known scenarios or synthetic trajectory subsets. |
| NFR-TEST-004 | Performance benchmarks (runtime per video minute; train time) shall be measurable. |
| NFR-TEST-005 | Automated tests shall run on CPU without GPU hardware or large dataset downloads. |

---

## 9. Documentation & Attribution

| ID | Requirement |
|----|-------------|
| NFR-DOC-001 | Architecture, APIs, schema, and roadmap shall live under `docs/`. |
| NFR-DOC-002 | Docs shall clearly separate UAV-OBB (training dataset) from Traffilytics (the platform). |
| NFR-DOC-003 | UAV-OBB shall be cited per CC BY 4.0 in the repository and wherever its imagery is displayed in the product. |
| NFR-DOC-004 | Third-party licence obligations of runtime dependencies (notably Ultralytics, AGPL-3.0) shall be documented and honoured by the project's own licence. |
