"""
Интеграционные тесты QdrantStore.
Требуют запущенный Qdrant на localhost:6333.
"""
import pytest
import pytest_asyncio

from app.rag.embedder import VECTOR_SIZE
from app.rag.qdrant_store import QdrantChunk, QdrantStore


TEST_COLLECTION = "test_knowledge"


@pytest_asyncio.fixture
async def store():
    """Подключённый QdrantStore с тестовой коллекцией, очисткой после теста."""
    s = QdrantStore()
    s._collection = TEST_COLLECTION
    await s.connect()
    client = s._get_client()
    try:
        await client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    await s.ensure_collection()
    yield s
    try:
        await client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    await s.disconnect()


def _make_fake_vector(seed: int) -> list[float]:
    """
    Сделать различимый нормализованный вектор для теста.

    Каждый вектор «торчит» в свою сторону: компонента №seed ставится в 1,
    остальные в 0. Это даёт seed-векторам ортогональность и чёткое
    превосходство при поиске самого себя.
    """
    vec = [0.0] * VECTOR_SIZE
    vec[seed % VECTOR_SIZE] = 1.0
    return vec


async def test_ensure_collection_idempotent(store: QdrantStore):
    """Повторный вызов ensure_collection не падает."""
    await store.ensure_collection()
    await store.ensure_collection()


async def test_upsert_and_search(store: QdrantStore):
    """Записываем 3 чанка разных типов, ищем — должны найтись."""
    chunks = [
        QdrantChunk(
            vector=_make_fake_vector(1),
            source_type="support",
            source_id="https://cdn.example.com/welcomebook.pdf",
            title="Велкомбук",
            text="Текст про отпуск",
            chunk_index=0,
        ),
        QdrantChunk(
            vector=_make_fake_vector(2),
            source_type="wiki_page",
            source_id="https://std.kitdev.ru/page1",
            title="Wiki страница",
            text="Текст про регламент",
            chunk_index=0,
        ),
        QdrantChunk(
            vector=_make_fake_vector(3),
            source_type="faq",
            source_id="faq://question/1",
            title="Вопрос про обед",
            text="Где находится столовая?",
            chunk_index=0,
        ),
    ]
    await store.upsert(chunks)

    results = await store.search(_make_fake_vector(1), top_k=3)
    assert len(results) == 3
    assert results[0].text == "Текст про отпуск"


async def test_search_with_source_type_filter(store: QdrantStore):
    """Фильтр по source_type оставляет только нужные."""
    await store.upsert([
        QdrantChunk(
            vector=_make_fake_vector(1),
            source_type="support",
            source_id="https://cdn.example.com/doc",
            title="Doc",
            text="doc text",
            chunk_index=0,
        ),
        QdrantChunk(
            vector=_make_fake_vector(2),
            source_type="wiki_page",
            source_id="https://std.kitdev.ru/page",
            title="Wiki",
            text="wiki text",
            chunk_index=0,
        ),
    ])

    results = await store.search(
        _make_fake_vector(1),
        top_k=10,
        source_types=["wiki_page"],
    )
    assert len(results) == 1
    assert results[0].source_type == "wiki_page"


async def test_search_with_multiple_source_types(store: QdrantStore):
    """Фильтр по нескольким source_type — например, искать в документах и FAQ, но не в wiki."""
    await store.upsert([
        QdrantChunk(
            vector=_make_fake_vector(1),
            source_type="support",
            source_id="https://cdn.example.com/welcome.pdf",
            title="Велкомбук",
            text="документ",
            chunk_index=0,
        ),
        QdrantChunk(
            vector=_make_fake_vector(2),
            source_type="faq",
            source_id="faq://1",
            title="FAQ",
            text="faq",
            chunk_index=0,
        ),
        QdrantChunk(
            vector=_make_fake_vector(3),
            source_type="wiki_page",
            source_id="https://std.kitdev.ru/page",
            title="Wiki",
            text="wiki",
            chunk_index=0,
        ),
    ])

    results = await store.search(
        _make_fake_vector(1),
        top_k=10,
        source_types=["support", "faq"],
    )
    assert len(results) == 2
    types_in_result = {r.source_type for r in results}
    assert types_in_result == {"support", "faq"}


async def test_upsert_same_id_overwrites(store: QdrantStore):
    """Повторный upsert того же (source_id, chunk_index) обновляет, а не дублирует."""
    chunk_v1 = QdrantChunk(
        vector=_make_fake_vector(1),
        source_type="regulation",
        source_id="https://cdn.example.com/same",
        title="V1",
        text="version 1",
        chunk_index=0,
    )
    chunk_v2 = QdrantChunk(
        vector=_make_fake_vector(1),
        source_type="regulation",
        source_id="https://cdn.example.com/same",
        title="V2",
        text="version 2",
        chunk_index=0,
    )
    await store.upsert([chunk_v1])
    await store.upsert([chunk_v2])

    results = await store.search(_make_fake_vector(1), top_k=10)
    assert len(results) == 1
    assert results[0].title == "V2"


async def test_delete_by_source(store: QdrantStore):
    """Удаление по source_id убирает все чанки этого документа."""
    await store.upsert([
        QdrantChunk(
            vector=_make_fake_vector(1),
            source_type="support",
            source_id="https://cdn.example.com/to-delete",
            title="To delete",
            text="part 1",
            chunk_index=0,
        ),
        QdrantChunk(
            vector=_make_fake_vector(2),
            source_type="support",
            source_id="https://cdn.example.com/to-delete",
            title="To delete",
            text="part 2",
            chunk_index=1,
        ),
        QdrantChunk(
            vector=_make_fake_vector(3),
            source_type="support",
            source_id="https://cdn.example.com/to-keep",
            title="To keep",
            text="keep me",
            chunk_index=0,
        ),
    ])

    await store.delete_by_source("https://cdn.example.com/to-delete")

    results = await store.search(_make_fake_vector(1), top_k=10)
    assert len(results) == 1
    assert results[0].source_id == "https://cdn.example.com/to-keep"


async def test_blank_source_type(store: QdrantStore):
    """Бланки тоже индексируются как один чанк с описанием."""
    await store.upsert([
        QdrantChunk(
            vector=_make_fake_vector(1),
            source_type="blank",
            source_id="https://cdn.example.com/vacation_blank.docx",
            title="Заявление на оплачиваемый отпуск",
            text="Заявление на оплачиваемый отпуск. Шаблон для подачи в отдел кадров.",
            chunk_index=0,
        ),
    ])

    results = await store.search(
        _make_fake_vector(1),
        top_k=10,
        source_types=["blank"],
    )
    assert len(results) == 1
    assert results[0].source_type == "blank"
    assert "Заявление" in results[0].title


async def test_upsert_empty_list_noop(store: QdrantStore):
    """Пустой список не падает."""
    await store.upsert([])


async def test_get_indexed_at_returns_value(store: QdrantStore):
    """get_indexed_at возвращает значение из payload."""
    await store.upsert([
        QdrantChunk(
            vector=_make_fake_vector(1),
            source_type="support",
            source_id="https://cdn.example.com/doc.pdf",
            title="Doc",
            text="text",
            chunk_index=0,
            indexed_at="2026-06-04T10:00:00",
        ),
    ])

    result = await store.get_indexed_at("https://cdn.example.com/doc.pdf")
    assert result == "2026-06-04T10:00:00"


async def test_get_indexed_at_returns_none_for_unknown_source(store: QdrantStore):
    """Если источника нет в Qdrant — возвращается None."""
    result = await store.get_indexed_at("https://cdn.example.com/never-indexed.pdf")
    assert result is None