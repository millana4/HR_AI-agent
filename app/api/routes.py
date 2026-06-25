from dataclasses import replace

from fastapi import APIRouter, Depends, Request

from app.api.schemas import AskRequest, AskResponse, HealthResponse, Alert
from app.core.config import Config
from app.core.exceptions import LLMError, LLMAuthError
from app.core.logging import get_logger
from app.services.agent_common import AgentLoop
from app.services.agent_loop_gigachat import process_request_gigachat
from app.services.agent_loop_yandex import process_request_yandex


router = APIRouter()
logger = get_logger(__name__)


def get_agent_loop(request: Request) -> AgentLoop:
    """Достать agent_loop из app.state (положили туда в lifespan)."""
    return request.app.state.agent_loop


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Проверка работоспособности сервиса."""
    return HealthResponse()


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    agent: AgentLoop = Depends(get_agent_loop),
) -> AskResponse:
    """Обработать запрос пользователя. Петля выбирается по LLM_PROVIDER."""
    correlation_id = payload.correlation_id or request.state.correlation_id

    logger.info(
        f"Ask request: user_id={payload.user_id} request_len={len(payload.request)}",
        extra={"correlation_id": correlation_id},
    )

# GigaChat как основной провайдер — сразу его петля, без fallback.
    if Config.LLM_PROVIDER.lower() != "yandex":
        return await process_request_gigachat(
            agent=agent,
            user_id=payload.user_id,
            request_text=payload.request,
            correlation_id=correlation_id,
        )

    # Yandex как основной. При ЛЮБОЙ ошибке LLM — fallback на GigaChat-петлю.
    try:
        return await process_request_yandex(
            agent=agent,
            user_id=payload.user_id,
            request_text=payload.request,
            correlation_id=correlation_id,
        )
    except LLMError as exc:
        # LLMAuthError (401/403) — подкласс LLMError; только на него шлём алерт.
        is_auth = isinstance(exc, LLMAuthError)
        logger.warning(
            f"Yandex LLM failed ({'auth/payment' if is_auth else 'temporary'}): "
            f"{exc} — fallback на GigaChat",
            extra={"correlation_id": correlation_id},
        )
        if agent.fallback_llm is None:
            logger.error(
                "Fallback недоступен (fallback_llm=None) — пробрасываем ошибку",
                extra={"correlation_id": correlation_id},
            )
            raise

        # Подменяем основной клиент запасным и идём через GigaChat-петлю.
        fallback_agent = replace(agent, llm=agent.fallback_llm)
        response = await process_request_gigachat(
            agent=fallback_agent,
            user_id=payload.user_id,
            request_text=payload.request,
            correlation_id=correlation_id,
        )

        # Только на auth-сбой Yandex (401/403) — алерт админам.
        if is_auth:
            response.alert = Alert(
                type="provider_switch",
                message=(
                    "Yandex AI Studio отверг запрос (HTTP 401/403): невалидный "
                    "ключ или закончились средства. Агент временно работает на "
                    "запасном провайдере (GigaChat)."
                ),
            )
            logger.info(
                "Alert сформирован: provider_switch (Yandex auth/payment)",
                extra={"correlation_id": correlation_id},
            )

        return response