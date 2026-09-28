"""Engine, session factory, and schema bootstrap."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import Base


class Database:
    """SQLite (or other SQLAlchemy URL) store with a write lock for SQLite threads."""

    def __init__(self, url: str) -> None:
        connect_args: dict[str, object] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            self._ensure_sqlite_parent(url)
        self.engine: Engine = create_engine(
            url,
            future=True,
            connect_args=connect_args,
        )
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        self.write_lock = threading.RLock()

    @staticmethod
    def _ensure_sqlite_parent(url: str) -> None:
        if url in {"sqlite://", "sqlite:///:memory:", "sqlite://"}:
            return
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            return
        path = Path(url[len(prefix) :])
        if path.parent and str(path.parent) not in {".", ""}:
            path.parent.mkdir(parents=True, exist_ok=True)

    def create_all(self) -> None:
        """Create tables if they do not exist."""
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._session_factory()

    @contextmanager
    def session_scope(self, *, write: bool = False) -> Iterator[Session]:
        """Yield a session; commit when ``write`` is True."""
        session = self.session()
        try:
            if write:
                with self.write_lock:
                    yield session
                    session.commit()
            else:
                yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
