"""FastAPI application factory (FR-API-001)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.errors import http_exception_handler, validation_exception_handler
from backend.api.routes import analytics as analytics_routes
from backend.api.routes import events as events_routes
from backend.api.routes import jobs as jobs_routes
from backend.api.routes import models as models_routes
from backend.api.routes import videos as videos_routes
from backend.database.session import Database
from backend.services.jobs import JobBroker
from backend.services.pipeline import PipelineWorker
from backend.services.settings import Settings, load_settings


def create_app(
    settings: Settings | None = None,
    *,
    db: Database | None = None,
    broker: JobBroker | None = None,
    start_worker: bool = True,
) -> FastAPI:
    """Build the API + dashboard app.

    Tests pass a temp ``Settings`` and may inject a ``Database`` / ``JobBroker``.
    Production uses ``create_app()`` (uvicorn ``backend.api.app:app``).
    """
    cfg = settings or load_settings()
    database = db or Database(cfg.db_url)
    database.create_all()
    worker = PipelineWorker(database, cfg)
    job_broker = broker or JobBroker(worker.run_job)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            job_broker.start()
        yield
        if start_worker:
            job_broker.stop()

    app = FastAPI(
        title="Traffilytics API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.db = database
    app.state.broker = job_broker
    app.state.worker = worker

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    app.include_router(videos_routes.router, prefix="/api/v1")
    app.include_router(jobs_routes.router, prefix="/api/v1")
    app.include_router(analytics_routes.router, prefix="/api/v1")
    app.include_router(events_routes.router, prefix="/api/v1")
    app.include_router(models_routes.router, prefix="/api/v1")

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    _mount_frontend(app, cfg)
    return app


def _mount_frontend(app: FastAPI, cfg: Settings) -> None:
    """Serve the Vite SPA. Client routes fall back to index.html (not Next.js)."""
    root = Path(cfg.frontend_dir)
    dist = root / "dist"
    index = dist / "index.html" if (dist / "index.html").is_file() else root / "index.html"
    if not index.is_file():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="frontend-assets")

    @app.get("/")
    def dashboard_root() -> FileResponse:
        return FileResponse(index)

    @app.get("/{full_path:path}")
    def dashboard_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api") or full_path.split("/", 1)[0] in {
            "docs",
            "redoc",
            "openapi.json",
        }:
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = dist / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


def __getattr__(name: str):
    """Lazily build ``app`` so ``import create_app`` does not open the product DB."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
