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


setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Действия при старте и остановке приложения."""
    logger.info("AI Agent service started")
    yield
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