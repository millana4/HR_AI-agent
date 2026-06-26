"""
Тесты search_internal (единственный agent_internal tool).

Используем QdrantStore поверх in-memory клиента qdrant-client (:memory:) —
это настоящая реализация upsert/search, без Docker и без сети.
Embedder мокаем: подставляем фиксированные ортогональные векторы.

ВАЖНО (изменения под Yandex-ветку):
- execute_internal_tool теперь возвращает КОРТЕЖ (context, hidden_data_list).
- Hidden_data БОЛЬШЕ НЕ кладётся в контекст (реальные значения скрыты от
  нейронки). В тексте остаётся плейсхолдер #ИМЯ, а сами значения
  возвращаются отдельным списком hidden_data_list для программной
  подстановки после Pass 2.

Замечание про in-memory Qdrant: при ортогональных фейковых векторах поиск
возвращает ВСЕ чанки (несовпадающие — со score 0, но в пределах top_k).
Поэтому проверяем не «список ровно пуст», а «значение не утекло в контекст».
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
            # В тексте — ПЛЕЙСХОЛДЕР с решёткой, не реальное значение.
            text="Вопрос: Кто директор? Ответ: Обратитесь к #ДИРЕКТОР.",
            chunk_index=0,
            link="https://wiki.example.com/dir",
            # Реальное значение лежит в hidden_data (отдаётся отдельно).
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


async def test_finds_faq_with_placeholder_and_link(populated_store):
    """
    FAQ-чанк: в контексте остаётся плейсхолдер #ДИРЕКТОР (НЕ реальное имя),
    а само значение возвращается в hidden_data_list. Ссылка — в контексте.
    """
    context, hidden_data_list = await execute_internal_tool(
        "search_internal", {"query": "кто директор"},
        populated_store, _embedder(1), correlation_id="t",
    )
    # В контексте — плейсхолдер, а НЕ реальное значение.
    assert "#ДИРЕКТОР" in context
    assert "Иванов Иван" not in context
    # Ссылка остаётся в контексте.
    assert "https://wiki.example.com/dir" in context
    # Реальное значение — в отдельном списке hidden_data.
    assert "ДИРЕКТОР=Иванов Иван" in hidden_data_list


async def test_finds_document_with_cdn_url(populated_store):
    """Бланк находится, его URL — в контексте. Реальное значение скрытых
    данных чужого faq-чанка (in-memory Qdrant возвращает все чанки) в
    контекст НЕ утекает."""
    context, hidden_data_list = await execute_internal_tool(
        "search_internal", {"query": "бланк отпуска"},
        populated_store, _embedder(2), correlation_id="t",
    )
    assert "Заявление на отпуск" in context
    assert "https://cdn.example.com/vacation.docx" in context
    # Главное: реальное значение скрытых данных не попало в контекст.
    assert "Иванов Иван" not in context


async def test_finds_wiki_with_url(populated_store):
    context, hidden_data_list = await execute_internal_tool(
        "search_internal", {"query": "ЭДО"},
        populated_store, _embedder(3), correlation_id="t",
    )
    assert "Подключение к ЭДО" in context
    assert "https://std.kitdev.ru/edo" in context


async def test_searches_all_sources_at_once(populated_store):
    """search_internal ищет по всем источникам — embed_query вызывается один раз."""
    emb = _embedder(1)
    await execute_internal_tool(
        "search_internal", {"query": "что угодно"},
        populated_store, emb, correlation_id="t",
    )
    emb.embed_query.assert_awaited_once()


async def test_empty_query_returns_no_context(populated_store):
    """Пустой query → (NO_CONTEXT, [])."""
    context, hidden_data_list = await execute_internal_tool(
        "search_internal", {"query": "   "},
        populated_store, _embedder(1), correlation_id="t",
    )
    assert context == NO_CONTEXT
    assert hidden_data_list == []


async def test_unknown_tool_returns_no_context(populated_store):
    """Неизвестный tool → (NO_CONTEXT, [])."""
    context, hidden_data_list = await execute_internal_tool(
        "search_unknown", {"query": "x"},
        populated_store, _embedder(1), correlation_id="t",
    )
    assert context == NO_CONTEXT
    assert hidden_data_list == []


async def test_no_results_returns_no_context(store: QdrantStore):
    """Пустой store → (NO_CONTEXT, [])."""
    context, hidden_data_list = await execute_internal_tool(
        "search_internal", {"query": "что угодно"},
        store, _embedder(1), correlation_id="t",
    )
    assert context == NO_CONTEXT
    assert hidden_data_list == []


async def test_hidden_data_collected_and_not_in_context(store: QdrantStore):
    """Hidden_data собирается со всех faq-чанков, но значения не утекают в контекст."""
    await store.upsert([
        QdrantChunk(
            vector=_vec(1),
            source_type="faq",
            source_id="nocodb_id:1",
            title="A",
            text="Ответ: пишите на #КОНТАКТ_A.",
            chunk_index=0,
            hidden_data="КОНТАКТ_A=a@mavis.ru",
        ),
        QdrantChunk(
            vector=_vec(2),
            source_type="faq",
            source_id="nocodb_id:2",
            title="B",
            text="Ответ: пишите на #КОНТАКТ_B.",
            chunk_index=0,
            hidden_data="КОНТАКТ_B=b@mavis.ru",
        ),
    ])
    context, hidden_data_list = await execute_internal_tool(
        "search_internal", {"query": "контакты"},
        store, _embedder(1), correlation_id="t",
    )
    # Оба значения собраны в список.
    assert "КОНТАКТ_A=a@mavis.ru" in hidden_data_list
    assert "КОНТАКТ_B=b@mavis.ru" in hidden_data_list
    # Но НИ одного реального email нет в контексте.
    assert "a@mavis.ru" not in context
    assert "b@mavis.ru" not in context
