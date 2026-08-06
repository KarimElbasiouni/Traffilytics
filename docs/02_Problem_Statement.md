# Problem Statement

## Context

Transportation researchers and traffic engineers need structured vehicle trajectories and multi-scale flow metrics from aerial video. Open datasets such as **DRIFT** provide high-quality 4K drone imagery, OBB annotations, and ground-truth trajectories for nine interconnected intersections in Daejeon—along with research scripts and a reference YOLOv11m + ByteTrack setup.

That still leaves a gap: a **complete, modular traffic intelligence platform** that trains its own detector, runs its own tracking and trajectory generation, implements its own analytics and event logic, stores results, exposes APIs, and presents interactive dashboards and reports. DRIFT is a dataset and reference codebase; it is not a deployable product.

## The Problem

There is no lightweight, end-to-end platform that:

1. **Owns** video ingestion, preprocessing, frame extraction, and metadata management
2. **Trains and evaluates** an OBB detector on DRIFT annotations (rather than only consuming a pre-shipped checkpoint as the product)
3. **Integrates and evaluates** multi-object tracking and **generates** trajectories from that pipeline
4. **Implements** custom analytics (flow, bottlenecks, imbalance, events) and automated insights
5. **Persists** results and serves them through a backend API and interactive dashboard
6. **Packages** the system as deployable software (e.g., Dockerized services)

Relying on DRIFT’s provided trajectory CSVs as the live data path would skip the computer vision pipeline that the platform is meant to demonstrate. Those CSVs are for **validation and benchmarking**, not as a substitute for generated tracks.

## Who Is Affected

| Stakeholder | Pain |
|-------------|------|
| Traffic / transportation engineers | Research repos don’t offer a unified dashboard + reports workflow |
| Researchers & students | Hard to go from training on DRIFT → own trajectories → product-style analytics |
| Future operators | Need a modular platform pattern, not only notebook demos |

## Opportunity

Use DRIFT as the primary **training and evaluation** dataset while building Traffilytics as the platform:

- Train a YOLO OBB model on DRIFT annotations; evaluate against held-out labels
- Integrate ByteTrack (and optionally compare OC-SORT / DeepSORT)
- Generate trajectories; benchmark against DRIFT ground-truth CSVs
- Design analytics, events, insights, FastAPI backend, database, and dashboard independently
- Reuse YOLO architecture, ByteTrack, OpenCV, and PyTorch rather than reinventing them
- Reuse DRIFT stabilized videos / stabilization process without making Stabilo a project focus

## Success Criteria (Problem Solved When)

- Footage flows through Traffilytics’ own ingest → detect → track → trajectory pipeline
- Detector and tracker performance can be evaluated using DRIFT annotations / GT trajectories
- Custom analytics, bottlenecks, imbalance, events, and insights are available via API and dashboard
- The system is structured as a modular, deployable platform—not a fork of DRIFT’s research repo

## Out of Scope for This Problem

Solving this problem does **not** require controlling signals, recommending infrastructure projects, predicting long-term urban growth, providing emergency dispatch, inventing a new detector/tracker architecture, or centering R&D on video stabilization.
