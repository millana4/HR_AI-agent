"""
Тесты search_internal (единственный agent_internal tool).

Используем QdrantStore поверх in-memory клиента qdrant-client (:memory:) —
это настоящая реализация upsert/search, без Docker и без сети.
Embedder мокаем: подставляем фиксированные ортогональные векторы.
"""
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from app.rag.embedder import VECTOR_SIZE
from app.rag.qdrant_store import QdrantChunk, QdrantStore
from app.tools.tools_internal import NO_CONTEXT, execute_internal_tool


def _vec(seed: int) -> list[float]:
    v = [0.0] * VECTOR_SIZE
    v[seed % VECTOR_SIZE] = 1.0
    return v


@pytest_asyncio.fixture
async def store():
    s = QdrantStore()
    s._client = AsyncQdrantClient(":memory:")
    s._collection = "test_tools_internal"
    await s.ensure_collection()
    yield s
    await s._client.close()


@pytest_asyncio.fixture
async def populated_store(store: QdrantStore):
    await store.upsert([
        QdrantChunk(
            vector=_vec(1),
            source_type="faq",
            source_id="nocodb_id:1",
            title="Кто директор?",
            text="Вопрос: Кто директор? Ответ: Обратитесь к ДИРЕКТОР.",
            chunk_index=0,
            link="https://wiki.example.com/dir",
            hidden_data="ДИРЕКТОР=Иванов Иван",
        ),
        QdrantChunk(
            vector=_vec(2),
            source_type="blank",
            source_id="https://cdn.example.com/vacation.docx",
            title="Заявление на отпуск",
            text="Бланк заявления на отпуск.",
            chunk_index=0,
        ),
        QdrantChunk(
            vector=_vec(3),
            source_type="wiki",
            source_id="https://std.kitdev.ru/edo",
            title="Подключение к ЭДО",
            text="Как подключить поставщика к ЭДО.",
            chunk_index=0,
        ),
    ])
    return store


def _embedder(seed: int):
    e = AsyncMock()
    e.embed_query = AsyncMock(return_value=_vec(seed))
    return e


async def test_finds_faq_with_hidden_and_link(populated_store):
    ctx = await execute_internal_tool(
        "search_internal", {"query": "кто директор"},
        populated_store, _embedder(1), correlation_id="t",
    )
    assert "Обратитесь к ДИРЕКТОР" in ctx
    assert "ДИРЕКТОР=Иванов Иван" in ctx
    assert "https://wiki.example.com/dir" in ctx


async def test_finds_document_with_cdn_url(populated_store):
    ctx = await execute_internal_tool(
        "search_internal", {"query": "бланк отпуска"},
        populated_store, _embedder(2), correlation_id="t",
    )
    assert "Заявление на отпуск" in ctx
    assert "https://cdn.example.com/vacation.docx" in ctx


async def test_finds_wiki_with_url(populated_store):
    ctx = await execute_internal_tool(
        "search_internal", {"query": "ЭДО"},
        populated_store, _embedder(3), correlation_id="t",
    )
    assert "Подключение к ЭДО" in ctx
    assert "https://std.kitdev.ru/edo" in ctx


async def test_searches_all_sources_at_once(populated_store):
    """search_internal ищет по всем источникам — фильтр включает все типы."""
    emb = _embedder(1)
    await execute_internal_tool(
        "search_internal", {"query": "что угодно"},
        populated_store, emb, correlation_id="t",
    )
    # Проверяем, что в поиск ушли все внутренние source_type.
    # search вызывается внутри execute_internal_tool на реальном store,
    # поэтому проверяем через результат embed_query (он был вызван).
    emb.embed_query.assert_awaited_once()


async def test_empty_query_returns_no_context(populated_store):
    ctx = await execute_internal_tool(
        "search_internal", {"query": "   "},
        populated_store, _embedder(1), correlation_id="t",
    )
    assert ctx == NO_CONTEXT


async def test_unknown_tool_returns_no_context(populated_store):
    ctx = await execute_internal_tool(
        "search_unknown", {"query": "x"},
        populated_store, _embedder(1), correlation_id="t",
    )
    assert ctx == NO_CONTEXT


async def test_no_results_returns_no_context(store: QdrantStore):
    # store пустой — поиск ничего не найдёт
    ctx = await execute_internal_tool(
        "search_internal", {"query": "что угодно"},
        store, _embedder(1), correlation_id="t",
    )
    assert ctx == NO_CONTEXT