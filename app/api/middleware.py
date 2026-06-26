import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import Config
from app.core.logging import generate_correlation_id, get_logger


logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Извлекает correlation_id из заголовка X-Correlation-Id или генерирует новый.
    Кладёт в request.state.correlation_id для дальнейшего использования.
    Возвращает в заголовке ответа.
    """

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-Id")
        if not correlation_id:
            correlation_id = generate_correlation_id()

        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-Id"] = correlation_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Логирует каждый входящий запрос и ответ.
    Включает correlation_id из request.state.
    """

    async def dispatch(self, request: Request, call_next):
        correlation_id = getattr(request.state, "correlation_id", "-")
        start = time.perf_counter()

        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={"correlation_id": correlation_id},
        )

        response = await call_next(request)

        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Response: {response.status_code} in {duration_ms}ms",
            extra={"correlation_id": correlation_id},
        )

        return response


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Проверяет заголовок X-API-Key.
    Эндпоинты в EXEMPT_PATHS пропускаются без проверки.
    """

    EXEMPT_PATHS = {"/api/v1/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        correlation_id = getattr(request.state, "correlation_id", "-")
        provided_key = request.headers.get("X-API-Key")

        if not provided_key:
            logger.warning(
                "Missing X-API-Key",
                extra={"correlation_id": correlation_id},
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing X-API-Key",
                    "detail": "Header X-API-Key is required",
                    "correlation_id": correlation_id,
                },
            )

        if provided_key != Config.AI_AGENT_API_KEY:
            logger.warning(
                "Invalid X-API-Key",
                extra={"correlation_id": correlation_id},
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid X-API-Key",
                    "detail": "Provided X-API-Key is invalid",
                    "correlation_id": correlation_id,
                },
            )

        return await call_next(request)