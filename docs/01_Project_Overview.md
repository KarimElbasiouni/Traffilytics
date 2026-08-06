# Project Overview

## Project Name

**Traffilytics** — AI Traffic Intelligence Platform

## Description

Traffilytics is a complete traffic intelligence platform: modular video ingestion, a trained OBB detector, multi-object tracking, trajectory generation, a custom analytics engine, persistent storage, APIs, automated insights, and an interactive web dashboard.

The **[DRIFT open dataset](https://huggingface.co/datasets/Hj-Lee/The-DRIFT)** is the primary **training and evaluation** dataset—annotated 4K drone footage and ground-truth trajectories over nine interconnected urban intersections in Daejeon, South Korea. DRIFT’s repository is a dataset and reference implementation; Traffilytics is the product platform built around that data, not a thin wrapper of DRIFT scripts.

## DRIFT vs Traffilytics

| Component | DRIFT Repository | Traffilytics |
|-----------|------------------|--------------|
| Dataset | Annotated drone footage and trajectory data | Use DRIFT as primary training and evaluation dataset |
| Video processing | Basic preprocessing / frame extraction / stabilization scripts | Own modular ingestion, preprocessing, frame extraction, metadata management |
| Stabilization | Stabilo-based scripts | Reuse DRIFT stabilized videos or their process (**not a project focus**) |
| Object detection | Pre-trained YOLOv11m OBB (`best.pt`) | **Train and evaluate** your own YOLO OBB model on DRIFT annotations |
| Bounding boxes | Polygon-based OBB format | Train/deploy OBB detector using the same annotation format |
| Tracking | ByteTrack integration | Independently integrate and evaluate ByteTrack (or OC-SORT / DeepSORT) |
| Trajectories | Generated after their detect/track; GT CSVs provided | **Generate your own** from your detector + tracker |
| Trajectory CSVs | Ground-truth tracks | **Validation and benchmarking only** |
| Lane assignment | Included in dataset | Infer or utilize for analytics and visualization |
| Analytics | Example scripts (TTC, LC, congestion, flow–density, …) | Own analytics engine tailored to platform requirements |
| Flow / bottleneck / imbalance / events | Research examples or limited | Independently design and implement (imbalance as a dedicated module) |
| Insights | Not included | Automated human-readable summaries |
| Dashboard / API / DB / reports | Not provided | Full web dashboard, backend (e.g. FastAPI), analytics DB, automated reports |
| Deployment | Research codebase | Modular, deployable platform (e.g. Dockerized services) |

### What you are not reinventing

| Technology | Approach | Why |
|------------|----------|-----|
| YOLO architecture | Use existing YOLO; **train your own** OBB weights | Shows a real CV pipeline without inventing a new detector |
| ByteTrack (or similar) | Integrate an existing tracker | Contribution is pipeline integration and evaluation |
| OpenCV | Video processing | Standard CV library |
| PyTorch | Model training | Industry-standard DL framework |

## Primary Goals

1. Ingest and preprocess DRIFT (and compatible) traffic video with your own pipeline
2. Train and evaluate a YOLO OBB detector on DRIFT annotations
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
- Ship DRIFT’s provided trajectory CSVs as the production trajectory source

## MVP Definition

The MVP is complete when a user can:

1. Load DRIFT traffic footage through Traffilytics’ ingestion pipeline
2. Run **your** trained OBB detector and tracker on that footage
3. Obtain **generated** trajectories (benchmarked against DRIFT GT where useful)
4. See traffic metrics, bottlenecks, events, and automated insights
5. Explore results on an interactive dashboard backed by API + database

## Final Project Statement

Traffilytics demonstrates how a full software platform—detection, tracking, analytics, storage, APIs, and dashboard—can turn drone traffic video into actionable transportation insights. DRIFT supplies the data and annotation format; Traffilytics owns the end-to-end product implementation.
