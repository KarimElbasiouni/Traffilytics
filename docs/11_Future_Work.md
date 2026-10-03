# Future Work

Work beyond the MVP, consistent with Traffilytics as a **full platform** and UAV-OBB as **training/evaluation data**.

---

## 1. Platform Expansion

- Additional OBB datasets beyond UAV-OBB (with separate adapters)
- Live or near-live camera/drone streams
- Continuous sliding-window analytics
- Multi-clip corridor dashboards
- Geo-aligned / orthophoto map overlays once calibration data is available

---

## 2. Insight Generation — LLM Integration (V2)

- LLM narratives from structured analytics JSON
- Audience-specific tones; multilingual summaries
- Metric/event citations inside generated text

---

## 3. Analytics Depth

- First-class LC / TTC product views
- Richer interactive flow–density and time–space exploration
- Automatic lane detection to replace manual polygon configuration
- Camera calibration workflow for reliable physical-unit speeds
- OD / turning movement estimates
- Signal-phase observation (not control)
- External context fusion (weather, incidents)

---

## 4. Model & Tracking Hardening

- Larger training sweeps and ablation on UAV-OBB splits
- Class-imbalance handling for the rarer classes (bike, other_vehicle, taxi)
- Systematic ByteTrack vs OC-SORT vs DeepSORT comparison reports
- Domain adaptation beyond the dataset's cities and conditions (glare, heavy occlusion, unseen viewpoints)
- Own annotated trajectory subset to enable true MOTA/IDF1 benchmarking
- GPU workers, batching, multi-clip queues

---

## 5. Product & Deployment

- Auth, roles, multi-tenant spaces
- Upload quotas, retention policies, and deletion for user video
- Export (CSV, GeoJSON, PDF)
- Alert webhooks (not emergency dispatch)
- Observability and audit logs on the single-process host. Containers stay out of scope

---

## 6. Evaluation & Research

- Public reporting of Traffilytics detector/tracker metrics on UAV-OBB
- Human-in-the-loop correction UI
- Papers/blogs that credit UAV-OBB as the dataset and Traffilytics as the platform

---

## Still Out of Scope (Unless Goals Change)

- Traffic signal control
- Construction project recommendations
- Replacing transportation engineers
- Long-term urban development prediction
- Real-time emergency response
- Autonomous driving
- Inventing new detector/tracker architectures
- Making video stabilization a core R&D pillar
- Reporting physical-unit metrics for uncalibrated video

---

## Prioritization Hint

| Priority | Theme |
|----------|--------|
| Near-term after MVP | Tracker comparison write-up, dashboard polish, lane editor UX, public single-process host |
| Medium-term | LLM insights, exports/alerts, deeper micro analytics, calibration workflow |
| Long-term | Live feeds, multi-dataset adapters, geo maps, production multi-tenant ops |

---

## References

- UAV-OBB dataset: https://data.mendeley.com/datasets/6snrjwcpkh/3
- UAV-OBB Kaggle mirror: https://www.kaggle.com/datasets/mdferozahmedafm/uav-obb-drone-based-urban-vehicle-dataset
- UAV-OBB paper (Data in Brief): https://doi.org/10.1016/j.dib.2026.112710
- Ultralytics YOLO (AGPL-3.0): https://github.com/ultralytics/ultralytics
- ByteTrack: https://github.com/ifzhang/ByteTrack
