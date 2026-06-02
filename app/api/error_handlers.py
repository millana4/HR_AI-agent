"""
Обработчики исключений: приводят ошибки разных типов к формату ErrorResponse.
"""
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import (
    AgentError,
    AuthError,
    LLMError,
    LLMTimeoutError,
    RepositoryError,
    ToolExecutionError,
)
from app.core.logging import get_logger


logger = get_logger(__name__)


def _get_correlation_id(request: Request) -> str:
    """Достаёт correlation_id из request.state. Если нет — прочерк."""
    return getattr(request.state, "correlation_id", "-")


def _make_error_response(
    status_code: int,
    error: str,
    detail: str | None,
    correlation_id: str,
) -> JSONResponse:
    """Формирует JSON-ответ в формате ErrorResponse."""
    content = {
        "error": error,
        "correlation_id": correlation_id,
    }
    if detail:
        content["detail"] = detail
    return JSONResponse(status_code=status_code, content=content)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Pydantic-валидация запроса не прошла."""
    correlation_id = _get_correlation_id(request)
    # Берём первую ошибку валидации — обычно её достаточно
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = ".".join(str(p) for p in first.get("loc", [])[1:])  # без "body"
        detail = f"Field '{loc}': {first.get('msg', 'validation failed')}"
    else:
        detail = "Validation failed"

    logger.warning(
        f"Validation error: {detail}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=400,
        error="Validation error",
        detail=detail,
        correlation_id=correlation_id,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Стандартные HTTPException от FastAPI (404, 405 и т.д.)."""
    correlation_id = _get_correlation_id(request)
    return _make_error_response(
        status_code=exc.status_code,
        error=str(exc.detail) if exc.detail else "HTTP error",
        detail=None,
        correlation_id=correlation_id,
    )


async def auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    """Ошибки авторизации в коде (не в middleware)."""
    correlation_id = _get_correlation_id(request)
    logger.warning(
        f"Auth error: {exc}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=401,
        error="Authentication error",
        detail=str(exc),
        correlation_id=correlation_id,
    )


async def llm_timeout_handler(
    request: Request,
    exc: LLMTimeoutError,
) -> JSONResponse:
    """LLM не ответила за таймаут."""
    correlation_id = _get_correlation_id(request)
    logger.error(
        f"LLM timeout: {exc}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=504,
        error="LLM timeout",
        detail=str(exc),
        correlation_id=correlation_id,
    )


async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    """Прочие ошибки LLM."""
    correlation_id = _get_correlation_id(request)
    logger.error(
        f"LLM error: {exc}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=502,
        error="LLM error",
        detail=str(exc),
        correlation_id=correlation_id,
    )


async def tool_execution_handler(
    request: Request,
    exc: ToolExecutionError,
) -> JSONResponse:
    """Ошибка выполнения tool."""
    correlation_id = _get_correlation_id(request)
    logger.error(
        f"Tool execution error: {exc}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=500,
        error="Tool execution error",
        detail=str(exc),
        correlation_id=correlation_id,
    )


async def repository_error_handler(
    request: Request,
    exc: RepositoryError,
) -> JSONResponse:
    """Ошибка работы с внешним хранилищем."""
    correlation_id = _get_correlation_id(request)
    logger.error(
        f"Repository error: {exc}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=500,
        error="Repository error",
        detail=str(exc),
        correlation_id=correlation_id,
    )


async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    """Базовый агентский exception — последний fallback."""
    correlation_id = _get_correlation_id(request)
    logger.error(
        f"Agent error: {exc}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=500,
        error="Internal error",
        detail=str(exc),
        correlation_id=correlation_id,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Всё, что не поймали — пишем в логи stacktrace и возвращаем 500."""
    correlation_id = _get_correlation_id(request)
    logger.exception(
        f"Unhandled exception: {exc}",
        extra={"correlation_id": correlation_id},
    )
    return _make_error_response(
        status_code=500,
        error="Internal server error",
        detail="An unexpected error occurred",
        correlation_id=correlation_id,
    )