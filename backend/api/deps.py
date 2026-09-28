"""FastAPI dependencies: database, settings, job broker."""

from __future__ import annotations

from fastapi import Request

from backend.database.session import Database
from backend.services.jobs import JobBroker
from backend.services.settings import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_broker(request: Request) -> JobBroker:
    return request.app.state.broker
