from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
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
from app.core.config import Config
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
from app.rag.embedder import get_embedder
from app.rag.qdrant_store import QdrantStore
from app.repositories.nocodb_client import NocoDBClient
from app.services.address_cache import get_address_cache
from app.services.agent_loop import AgentLoop
from app.services.departments_cache import DepartmentsCache
from app.services.pii_parser import PiiParser
from app.services.session_store import SessionStore


setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Старт: создаём долгоживущие клиенты и собираем AgentLoop в app.state.
    Остановка: корректно закрываем сессии (Redis, httpx, Qdrant).
    """
    logger.info("AI Agent service starting...")

    # 1. NocoDB-клиент — нужен PiiParser для загрузки кэша имён.
    nocodb_client = NocoDBClient()

    # 2. SessionStore — Redis для истории сессий.
    session_store = SessionStore()
    await session_store.connect()

    # 3. Основной LLM-клиент по конфигу (Yandex).
    llm_client = get_llm_client()

    #     Запасной клиент GigaChat для fallback — создаём, только если
    #     основной провайдер не GigaChat (иначе незачем дублировать).
    fallback_llm = None
    if Config.LLM_PROVIDER.lower() != "gigachat":
        fallback_llm = get_llm_client(provider="gigachat")

    # 4. PiiParser — заранее прогреваем кэш PII-имён из NocoDB.
    pii_parser = PiiParser()
    await pii_parser.ensure_ready(nocodb_client, correlation_id="startup")

    # 5. Qdrant — векторное хранилище для RAG-поиска.
    qdrant_store = QdrantStore()
    await qdrant_store.connect()
    await qdrant_store.ensure_collection(correlation_id="startup")

    # 6. Embedder — модель multilingual-e5-large. Прогреваем заранее,
    #    чтобы первый запрос пользователя не ждал загрузку ~2 ГБ модели.
    embedder = get_embedder()
    await embedder.embed_query("прогрев модели", correlation_id="startup")

    # 7. DepartmentsCache — списки отделов Мавис/Вотоня для промпта (кеш в Redis).
    departments_cache = DepartmentsCache()
    await departments_cache.connect()

    # 8. AddressCache — словарь адресов магазинов/аптек для каскадной
    #     классификации (извлечение топонима в query). Прогреваем при старте.
    address_cache = get_address_cache()
    await address_cache.ensure_fresh(nocodb_client, correlation_id="startup")

    # 9. Собираем AgentLoop и кладём в app.state.
    app.state.agent_loop = AgentLoop(
        llm=llm_client,
        session_store=session_store,
        pii_parser=pii_parser,
        nocodb_client=nocodb_client,
        qdrant_store=qdrant_store,
        embedder=embedder,
        departments_cache=departments_cache,
        address_cache=address_cache,
        fallback_llm=fallback_llm,
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

        if fallback_llm is not None:
            try:
                await fallback_llm.close()
            except Exception as exc:
                logger.warning(f"Error closing fallback LLM client: {exc}")

        try:
            await session_store.disconnect()
        except Exception as exc:
            logger.warning(f"Error disconnecting Redis: {exc}")

        try:
            await departments_cache.disconnect()
        except Exception as exc:
            logger.warning(f"Error disconnecting DepartmentsCache: {exc}")

        try:
            await qdrant_store.disconnect()
        except Exception as exc:
            logger.warning(f"Error disconnecting Qdrant: {exc}")

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

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    # Применяем ко всем путям, кроме /api/health.
    for path, methods in openapi_schema["paths"].items():
        if path.endswith("/health"):
            continue
        for method in methods.values():
            method["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = openapi_schema
    return openapi_schema


app.openapi = custom_openapi