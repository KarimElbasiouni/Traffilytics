# Future Work

Work beyond the MVP, consistent with Traffilytics as a **full platform** and DRIFT as **training/evaluation data**.

---

## 1. Platform Expansion

- Additional datasets beyond DRIFT (with separate adapters)
- Live or near-live camera/drone streams
- Continuous sliding-window analytics
- Multi-site corridor dashboards across all DRIFT intersections
- Geo-aligned / orthophoto map overlays when calibration data is used

---

## 2. Insight Generation — LLM Integration (V2)

- LLM narratives from structured analytics JSON
- Audience-specific tones; multilingual summaries
- Metric/event citations inside generated text

---

## 3. Analytics Depth

- First-class LC / TTC product views
- Richer interactive flow–density and time–space exploration
- OD / turning movement estimates across connected sites
- Signal-phase observation (not control)
- External context fusion (weather, incidents)

---

## 4. Model & Tracking Hardening

- Larger training sweeps and ablation on DRIFT splits
- Systematic ByteTrack vs OC-SORT vs DeepSORT comparison reports
- Domain adaptation (night, rain, glare, occlusion)
- GPU workers, batching, multi-clip queues

---

## 5. Product & Deployment

- Auth, roles, multi-tenant spaces
- Export (CSV, GeoJSON, PDF)
- Alert webhooks (not emergency dispatch)
- Production-grade Docker/K8s packaging, observability, audit logs

---

## 6. Evaluation & Research

- Public reporting of Traffilytics detector/tracker metrics on DRIFT
- Human-in-the-loop correction UI
- Papers/blogs that clearly credit DRIFT as dataset/reference, Traffilytics as platform

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
- Using DRIFT GT trajectory CSVs as the live product data path

---

## Prioritization Hint

| Priority | Theme |
|----------|--------|
| Near-term after MVP | Multi-site coverage, tracker comparison write-up, dashboard polish, Docker hardening |
| Medium-term | LLM insights, exports/alerts, deeper micro analytics |
| Long-term | Live feeds, multi-dataset adapters, geo maps, production multi-tenant ops |

---

## References

- DRIFT dataset: https://huggingface.co/datasets/Hj-Lee/The-DRIFT  
- DRIFT GitHub (reference): https://github.com/AIxMobility/The-DRIFT  
- DRIFT paper: https://arxiv.org/abs/2504.11019  
- Stabilo (if used): https://github.com/rfonod/stabilo  
