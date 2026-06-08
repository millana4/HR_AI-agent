from fastapi import APIRouter, Depends, Request

from app.api.schemas import AskRequest, AskResponse, HealthResponse
from app.core.logging import get_logger
from app.services.agent_loop import AgentLoop, process_request


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
    """Обработать запрос пользователя через двухпроходный agent_loop."""
    correlation_id = payload.correlation_id or request.state.correlation_id

    logger.info(
        f"Ask request: user_id={payload.user_id} request_len={len(payload.request)}",
        extra={"correlation_id": correlation_id},
    )

    return await process_request(
        agent=agent,
        user_id=payload.user_id,
        request_text=payload.request,
        correlation_id=correlation_id,
    )