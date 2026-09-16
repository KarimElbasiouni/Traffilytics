# Project Overview

## Project Name

**Traffilytics** — AI Traffic Intelligence Platform

## Description

Traffilytics is a complete traffic intelligence platform: modular video ingestion, a trained OBB detector, multi-object tracking, trajectory generation, a custom analytics engine, persistent storage, APIs, automated insights, and an interactive web dashboard.

The **[UAV-OBB open dataset](https://data.mendeley.com/datasets/6snrjwcpkh/3)** is the primary **training and evaluation** dataset — 1920×1080 nadir UAV imagery over urban roads in Chongqing and Wuhan, annotated with oriented bounding boxes across six vehicle classes. At runtime the platform processes **video the user uploads**.

UAV-OBB is a detection dataset and nothing more: still images and rotated boxes. Traffilytics supplies everything the dataset does not — identity across frames, trajectories, lane context, analytics, storage, APIs, and a dashboard.

## Dataset vs Platform

| Concern | UAV-OBB (dataset) | Traffilytics (platform) |
|---------|-------------------|-------------------------|
| Imagery | Nadir UAV stills at 1920×1080, plus a few supplementary MP4 clips | Own modular ingestion, preprocessing, frame extraction, metadata for user-uploaded video |
| Annotations | YOLOv8-OBB corner labels, six vehicle classes | **Trains and evaluates** its own YOLO OBB weights on them |
| Detection model | None shipped | Traffilytics-trained OBB weights are the product model |
| Tracking | None | Independently integrate and evaluate ByteTrack (or OC-SORT / DeepSORT) |
| Trajectories | None | **Generated** from own detector + tracker |
| Lane topology | None | User-defined lane/zone polygons per video |
| World scale | None | Optional pixel-to-metre scale; pixel-based units when absent |
| Analytics | None | Own engine: flow, density, bottleneck, imbalance, events |
| Insights | None | Automated human-readable summaries |
| Dashboard / API / DB / reports | None | Full web dashboard, backend (e.g. FastAPI), analytics DB, automated reports |
| Deployment | Not applicable | Modular, deployable platform (e.g. Dockerized services) |

### What you are not reinventing

| Technology | Approach | Why |
|------------|----------|-----|
| YOLO architecture | Use existing YOLO OBB; **train your own** weights | Shows a real CV pipeline without inventing a new detector |
| ByteTrack (or similar) | Integrate an existing tracker | Contribution is pipeline integration and evaluation |
| OpenCV | Video processing | Standard CV library |
| PyTorch / Ultralytics | Model training | Industry-standard tooling |

## Primary Goals

1. Ingest and preprocess user-uploaded aerial traffic video with your own pipeline
2. Train and evaluate a YOLO OBB detector on UAV-OBB annotations
3. Integrate and evaluate multi-object tracking (ByteTrack primary)
4. Generate trajectories from **your** detector + tracker
5. Implement a custom traffic analytics engine (flow, density, bottlenecks, imbalance, events)
6. Generate automated transportation insights
7. Persist results and expose them via backend APIs
8. Present findings through an interactive web dashboard and reports
9. Package the system as a modular, deployable platform

## Non-Goals

The system will **not**:

- Control traffic lights
- Recommend construction projects
- Replace transportation engineers
- Predict long-term urban development
- Provide real-time emergency response
- Perform autonomous driving functions
- Treat video stabilization as a core R&D focus
- Produce or curate a ground-truth trajectory dataset

## MVP Definition

The MVP is complete when a user can:

1. Upload aerial traffic footage and have Traffilytics ingest it
2. Run **your** trained OBB detector and tracker on that footage as a background job
3. Obtain **generated** trajectories with lane context from configured polygons
4. See traffic metrics, bottlenecks, events, and automated insights
5. Explore results on an interactive dashboard backed by API + database

## Final Project Statement

Traffilytics demonstrates how a full software platform — detection, tracking, analytics, storage, APIs, and dashboard — can turn drone traffic video into actionable transportation insights. UAV-OBB supplies the training data and annotation format; Traffilytics owns the end-to-end product implementation.
