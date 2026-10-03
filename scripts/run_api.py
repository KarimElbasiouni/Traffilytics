#!/usr/bin/env python3
"""Run the Traffilytics FastAPI app + dashboard (Epic 5).

    python scripts/run_api.py
    python scripts/run_api.py --host 127.0.0.1 --port 8000

Dashboard: http://127.0.0.1:8000/
API docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Traffilytics API and dashboard")
    # A platform sets PORT and expects the process on 0.0.0.0. Local runs stay on localhost.
    if os.environ.get("PORT") and not os.environ.get("HOST"):
        default_host = "0.0.0.0"
    else:
        default_host = os.environ.get("HOST", "127.0.0.1")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn is not installed. pip install -e '.[api]'",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
