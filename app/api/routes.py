from fastapi import APIRouter, Request

from app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    TextResponse,
)
from app.core.logging import get_logger


router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Проверка работоспособности сервиса."""
    return HealthResponse()


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """
    Заглушка. Реальная логика появится в шаге 13.
    """
    correlation_id = payload.correlation_id or request.state.correlation_id

    logger.info(
        f"Ask request: user_id={payload.user_id} request_len={len(payload.request)}",
        extra={"correlation_id": correlation_id},
    )

    return TextResponse(
        answer=f"Заглушка. Получен вопрос: {payload.request}",
        tool_used="answer_general",
        correlation_id=correlation_id,
    )