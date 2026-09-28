"""JSON error envelope used by every API route."""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=error_body(code, message))


class ApiError(HTTPException):
    """HTTPException that serializes to the project error envelope."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    code = getattr(exc, "code", None) or _status_code_name(exc.status_code)
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return error_response(exc.status_code, code, message)


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(422, "invalid_request", str(exc.errors()))


def _status_code_name(status: int) -> str:
    return {
        400: "invalid_request",
        404: "not_found",
        409: "conflict",
        501: "unsupported",
    }.get(status, "error")
