"""FastAPI TestClient coverage for Epic 5 (CPU, no YOLO)."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.database.persist import upsert_video
from backend.services.jobs import JobBroker
from backend.services.pipeline import PipelineWorker
from computer_vision.trajectories.generator import TrajectoryGenerator
from tests.backend.conftest import sample_trajectories, write_synthetic_mp4


@pytest.fixture
def client(settings, db):
    settings.skip_detect = True
    settings.skip_track = True
    worker = PipelineWorker(db, settings)
    broker = JobBroker(worker.run_job)
    app = create_app(settings, db=db, broker=broker, start_worker=True)
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_error_envelope(client) -> None:
    ok = client.get("/api/v1/health")
    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"
    missing = client.get("/api/v1/videos/nope")
    assert missing.status_code == 404
    body = missing.json()
    assert body["error"]["code"] == "not_found"
    assert "nope" in body["error"]["message"]


def test_dashboard_index(client) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Traffilytics" in res.text


def test_upload_returns_job_immediately(client, settings) -> None:
    video = write_synthetic_mp4(settings.raw_dir / "tmp_src.mp4")
    t0 = time.perf_counter()
    res = client.post(
        "/api/v1/videos",
        files={"file": ("site_03_clip.mp4", video.read_bytes(), "video/mp4")},
        data={"site": "lab"},
    )
    elapsed = time.perf_counter() - t0
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["status"] == "queued"
    assert body["job_id"].startswith("job_")
    assert body["video_id"]
    assert elapsed < 1.5
    job = client.get(f"/api/v1/jobs/{body['job_id']}")
    assert job.status_code == 200
    assert job.json()["status"] in {"queued", "processing", "failed", "completed"}


def test_query_endpoints_after_persist(client, db) -> None:
    from tests.backend.conftest import seed_completed_video

    seed_completed_video(db, "clip")
    overview = client.get("/api/v1/videos/clip/overview")
    assert overview.status_code == 200, overview.text
    data = overview.json()
    assert data["n_vehicles"] == 2
    assert data["units"]["labelled_as_physical"] is False
    assert "pixel-based" in data["units"]["note"]

    flow = client.get("/api/v1/videos/clip/analytics/flow")
    assert flow.status_code == 200
    assert "vehicles_per_minute" in flow.json()["flow"]

    fd = client.get("/api/v1/videos/clip/analytics/flow-density")
    assert fd.status_code == 200
    assert isinstance(fd.json()["flow_density"], list)

    btn = client.get("/api/v1/videos/clip/analytics/bottlenecks")
    assert btn.status_code == 200

    imb = client.get("/api/v1/videos/clip/analytics/imbalance")
    assert imb.status_code == 200

    heat = client.get("/api/v1/videos/clip/analytics/heatmap")
    assert heat.status_code == 200
    assert isinstance(heat.json()["heatmap"], list)

    events = client.get("/api/v1/videos/clip/events")
    assert events.status_code == 200

    insights = client.get("/api/v1/videos/clip/insights")
    assert insights.status_code == 200
    assert insights.json()["insights"]

    report = client.get("/api/v1/videos/clip/reports")
    assert report.status_code == 200
    payload = report.json()
    assert payload["findings"]
    assert "AGPL" in payload["attribution"]["licence"]

    vehicles = client.get("/api/v1/videos/clip/vehicles")
    assert vehicles.status_code == 200
    assert vehicles.json()["n_vehicles"] == 2

    traj = client.get("/api/v1/videos/clip/vehicles/1/trajectory")
    assert traj.status_code == 200
    assert traj.json()["points"]

    mapped = client.get("/api/v1/videos/clip/map")
    assert mapped.status_code == 200
    plan = mapped.json()
    assert plan["n_tracks"] == 2
    assert plan["tracks"][0]["points"]
    assert plan["bounds"]["x1"] >= plan["bounds"]["x0"]

    preview = client.get("/api/v1/videos/clip/preview")
    assert preview.status_code == 404

    micro = client.get("/api/v1/videos/clip/analytics/micro")
    assert micro.status_code == 200
    assert micro.json()["available"] is False


def test_put_lanes_writes_file_and_video_row(client, db, settings) -> None:
    with db.session_scope(write=True) as session:
        upsert_video(session, video_id="clip", status="queued", fps=10.0)
    body = {
        "lanes": [{"id": "lane_1", "polygon": [[0, 0], [64, 0], [64, 48], [0, 48]]}],
        "zones": [{"id": "zone_a", "polygon": [[0, 0], [32, 0], [32, 24], [0, 24]]}],
    }
    res = client.put("/api/v1/videos/clip/lanes", json=body)
    assert res.status_code == 200, res.text
    dest = settings.lanes_dir / "clip.json"
    assert dest.is_file()
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert saved["lanes"][0]["id"] == "lane_1"
    video = client.get("/api/v1/videos/clip")
    assert video.json()["lane_config"]["zones"][0]["id"] == "zone_a"


def test_frame_endpoint_serves_ingested_still(client, db, settings) -> None:
    from tests.backend.conftest import sample_trajectories, seed_completed_video
    from computer_vision.trajectories.generator import TrajectoryGenerator

    seed_completed_video(db, "clip")
    frames = settings.processed_dir / "clip" / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    still = np.zeros((48, 64, 3), dtype=np.uint8)
    still[:] = (40, 80, 120)
    assert cv2.imwrite(str(frames / "frame_000000.jpg"), still)
    assert cv2.imwrite(str(frames / "frame_000004.jpg"), still)
    TrajectoryGenerator().write_json(
        sample_trajectories("clip"),
        settings.processed_dir / "clip" / "trajectories.json",
        video_id="clip",
    )

    preview = client.get("/api/v1/videos/clip/preview")
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["n_frames"] == 2
    assert body["width"] == 64
    assert body["height"] == 48
    assert body["indices"] == [0, 4]

    image = client.get("/api/v1/videos/clip/frame")
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/")
    first = client.get("/api/v1/videos/clip/frame?i=0")
    assert first.status_code == 200
    missing = client.get("/api/v1/videos/clip/frame?i=9")
    assert missing.status_code == 404

    overlay = client.get("/api/v1/videos/clip/overlay")
    assert overlay.status_code == 200
    assert overlay.headers["content-type"].startswith("video/")
    assert len(overlay.content) > 0


def test_pipeline_ingest_analyze_persist(settings, db) -> None:
    video_id = "pipe_clip"
    raw = write_synthetic_mp4(settings.raw_dir / f"{video_id}.mp4")
    assert raw.is_file()
    processed = settings.processed_dir / video_id
    processed.mkdir(parents=True, exist_ok=True)
    trajectories = sample_trajectories(video_id)
    TrajectoryGenerator().write_json(trajectories, processed / "trajectories.json", video_id=video_id)

    with db.session_scope(write=True) as session:
        upsert_video(
            session,
            video_id=video_id,
            status="queued",
            source="upload",
            fps=10.0,
            resolution="64x48",
        )
    from backend.services.jobs import create_job

    job = create_job(db, video_id=video_id, options={"skip_detect": True, "skip_track": True})
    PipelineWorker(db, settings).run_job(job.job_id)

    from backend.database.models import Job, Report, Video

    with db.session_scope() as session:
        job_row = session.get(Job, job.job_id)
        assert job_row is not None
        assert job_row.status == "completed", job_row.error
        video = session.get(Video, video_id)
        assert video.status == "completed"
        report = session.get(Report, video_id)
        assert report is not None
        assert (processed / "analytics.json").is_file()
        assert (processed / "metadata.json").is_file()
