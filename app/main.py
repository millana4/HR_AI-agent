from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.error_handlers import (
    agent_error_handler,
    auth_error_handler,
    http_exception_handler,
    llm_error_handler,
    llm_timeout_handler,
    repository_error_handler,
    tool_execution_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.api.middleware import (
    ApiKeyAuthMiddleware,
    CorrelationIdMiddleware,
    LoggingMiddleware,
)
from app.api.routes import router
from app.core.exceptions import (
    AgentError,
    AuthError,
    LLMError,
    LLMTimeoutError,
    RepositoryError,
    ToolExecutionError,
)
from app.core.logging import get_logger, setup_logging
from app.llm.factory import get_llm_client
from app.repositories.nocodb_client import NocoDBClient
from app.services.agent_loop import AgentLoop
from app.services.pii_parser import PiiParser
from app.services.session_store import SessionStore


setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Старт: создаём долгоживущие клиенты и собираем AgentLoop в app.state.
    Остановка: корректно закрываем сессии (Redis, httpx).
    """
    logger.info("AI Agent service starting...")

    # 1. NocoDB-клиент — нужен PiiParser для загрузки кэша имён.
    nocodb_client = NocoDBClient()

    # 2. SessionStore — Redis для истории сессий.
    session_store = SessionStore()
    await session_store.connect()

    # 3. LLM-клиент по конфигу (сейчас — GigaChat).
    llm_client = get_llm_client()

    # 4. PiiParser — заранее прогреваем кэш PII-имён из NocoDB.
    pii_parser = PiiParser()
    await pii_parser.ensure_ready(nocodb_client, correlation_id="startup")

    # 5. Собираем AgentLoop и кладём в app.state.
    app.state.agent_loop = AgentLoop(
        llm=llm_client,
        session_store=session_store,
        pii_parser=pii_parser,
        nocodb_client=nocodb_client,
    )

    logger.info("AI Agent service started")

    try:
        yield
    finally:
        logger.info("AI Agent service stopping...")

        # Закрываем ресурсы в обратном порядке.
        try:
            await llm_client.close()
        except Exception as exc:
            logger.warning(f"Error closing LLM client: {exc}")

        try:
            await session_store.disconnect()
        except Exception as exc:
            logger.warning(f"Error disconnecting Redis: {exc}")

        try:
            await nocodb_client.close()
        except AttributeError:
            # У клиента нет метода close — игнорируем.
            pass
        except Exception as exc:
            logger.warning(f"Error closing NocoDB client: {exc}")

        logger.info("AI Agent service stopped")


app = FastAPI(
    title="HR AI Agent API",
    description="HTTP API для общения Telegram-бота с ИИ-агентом компании Мавис.",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware регистрируются в обратном порядке выполнения.
app.add_middleware(ApiKeyAuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(CorrelationIdMiddleware)

# Exception handlers — от частного к общему.
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(AuthError, auth_error_handler)
app.add_exception_handler(LLMTimeoutError, llm_timeout_handler)
app.add_exception_handler(LLMError, llm_error_handler)
app.add_exception_handler(ToolExecutionError, tool_execution_handler)
app.add_exception_handler(RepositoryError, repository_error_handler)
app.add_exception_handler(AgentError, agent_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(router, prefix="/api")