# Problem Statement

## Context

Transportation researchers and traffic engineers need structured vehicle trajectories and multi-scale flow metrics from aerial video. Open datasets such as **UAV-OBB** supply exactly the hard part of the perception problem — tight, rotation-aware oriented bounding boxes over real urban roads, across day, night, rain, and fog, at multiple zoom levels — but a folder of annotated still images is not a system.

## The Problem

There is no lightweight, end-to-end platform that:

1. **Owns** video ingestion, preprocessing, frame extraction, and metadata management
2. **Trains and evaluates** an OBB detector on public aerial annotations (rather than only consuming someone else's checkpoint)
3. **Integrates and evaluates** multi-object tracking and **generates** trajectories from that pipeline
4. **Implements** custom analytics (flow, bottlenecks, imbalance, events) and automated insights
5. **Persists** results and serves them through a backend API and interactive dashboard
6. **Packages** the system as deployable software (e.g., Dockerized services)

UAV-OBB sharpens the gap rather than closing it. The dataset has no trajectories, no lane topology, and no world-scale calibration — only per-image oriented boxes. Everything temporal and spatial has to be produced by the platform: tracking turns detections into trajectories, user-defined polygons turn pixels into lanes and zones, and an optional scale turns pixel motion into physical speed. That gap is precisely what Traffilytics builds.

## Who Is Affected

| Stakeholder | Pain |
|-------------|------|
| Traffic / transportation engineers | Detection datasets and research repos don’t offer a unified dashboard + reports workflow |
| Researchers & students | Hard to go from a public OBB dataset → trained model → own trajectories → product-style analytics |
| Anyone with drone footage | No simple way to upload a clip and get traffic metrics back |
| Future operators | Need a modular platform pattern, not only notebook demos |

## Opportunity

Use UAV-OBB as the **training and evaluation** dataset while building Traffilytics as the platform:

- Train a YOLO OBB model on UAV-OBB's six vehicle classes; evaluate against held-out labels
- Integrate ByteTrack (and optionally compare OC-SORT / DeepSORT)
- Generate trajectories, and assess their temporal stability on UAV-OBB's bundled video clips
- Design analytics, events, insights, FastAPI backend, database, and dashboard independently
- Accept **user-uploaded** footage at runtime, so the platform is not welded to one dataset
- Reuse YOLO, ByteTrack, OpenCV, and PyTorch rather than reinventing them

## Success Criteria (Problem Solved When)

- Uploaded footage flows through Traffilytics’ own ingest → detect → track → trajectory pipeline
- Detector accuracy is measurable on held-out UAV-OBB labels, and tracking stability is reviewable on video
- Custom analytics, bottlenecks, imbalance, events, and insights are available via API and dashboard
- Derived metrics state their units honestly, using pixel-based units when no scale is supplied
- The system is structured as a modular, deployable platform

## Out of Scope for This Problem

Solving this problem does **not** require controlling signals, recommending infrastructure projects, predicting long-term urban growth, providing emergency dispatch, inventing a new detector/tracker architecture, curating a ground-truth trajectory dataset, or centering R&D on video stabilization.
