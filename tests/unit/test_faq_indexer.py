"""Тесты FAQ-индексатора."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.indexing.faq_indexer import (
    _make_faq_source_id,
    _make_faq_text,
    _needs_reindex,
    index_faq,
)
from app.repositories.faq import FaqEntry


def _make_entry(
    id: int = 1,
    question: str = "Q",
    answer: str = "A",
    updated_at: datetime | None = None,
) -> FaqEntry:
    return FaqEntry(
        id=id, question=question, answer=answer,
        link=None, attachment=None, hidden_data=None,
        updated_at=updated_at or datetime(2026, 6, 4, 10, 0, 0, tzinfo=timezone.utc),
    )


def test_make_faq_source_id():
    assert _make_faq_source_id(1) == "nocodb_id:1"
    assert _make_faq_source_id(42) == "nocodb_id:42"


def test_make_faq_text():
    entry = _make_entry(question="Где столовая?", answer="На первом этаже")
    text = _make_faq_text(entry)
    assert "Вопрос: Где столовая?" in text
    assert "Ответ: На первом этаже" in text


def test_make_faq_text_excludes_hidden_data_and_link():
    """Hidden_data и Link НЕ должны попадать в текст для эмбеддинга."""
    entry = FaqEntry(
        id=1, question="Q", answer="A",
        link="https://x", attachment="https://y",
        hidden_data="АД_ДИР=Иван Иванов",
        updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    text = _make_faq_text(entry)
    assert "Иван Иванов" not in text
    assert "АД_ДИР" not in text
    assert "https://x" not in text


def test_needs_reindex_when_not_in_qdrant():
    """Если записи нет в Qdrant — индексируем."""
    assert _needs_reindex(
        datetime(2026, 6, 4, tzinfo=timezone.utc),
        None,
    ) is True


def test_needs_reindex_when_nocodb_newer():
    """UpdatedAt новее indexed_at → индексируем."""
    assert _needs_reindex(
        datetime(2026, 6, 5, tzinfo=timezone.utc),
        "2026-06-04T10:00:00+00:00",
    ) is True


def test_needs_reindex_when_indexed_recently():
    """indexed_at позже UpdatedAt → пропускаем."""
    assert _needs_reindex(
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        "2026-06-04T10:00:00+00:00",
    ) is False


def test_needs_reindex_when_updated_at_unknown():
    """Если UpdatedAt неизвестен — на всякий случай переиндексируем."""
    assert _needs_reindex(None, "2026-06-04T10:00:00+00:00") is True


@pytest.fixture
def mock_nocodb():
    return AsyncMock()


@pytest.fixture
def mock_qdrant():
    return AsyncMock()


@pytest.fixture
def mock_embedder():
    m = AsyncMock()
    m.embed_documents.return_value = [[0.1] * 1024]
    return m


async def test_index_faq_empty_list(mock_nocodb, mock_qdrant, mock_embedder, monkeypatch):
    """Если FAQ пустой — статистика нулевая, ничего не происходит."""
    async def fake_fetch(*args, **kwargs):
        return []

    monkeypatch.setattr("app.indexing.faq_indexer.fetch_faq", fake_fetch)

    stats = await index_faq(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 0
    assert stats.skipped == 0
    assert stats.errors == 0
    mock_qdrant.upsert.assert_not_called()


async def test_index_faq_indexes_new_entries(mock_nocodb, mock_qdrant, mock_embedder, monkeypatch):
    """Новые записи (не в Qdrant) индексируются."""
    entries = [_make_entry(id=1)]

    async def fake_fetch(*args, **kwargs):
        return entries

    monkeypatch.setattr("app.indexing.faq_indexer.fetch_faq", fake_fetch)
    mock_qdrant.get_indexed_at.return_value = None

    stats = await index_faq(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 1
    assert stats.skipped == 0
    mock_qdrant.upsert.assert_called_once()


async def test_index_faq_skips_unchanged(mock_nocodb, mock_qdrant, mock_embedder, monkeypatch):
    """Если UpdatedAt < indexed_at — пропускаем."""
    entries = [_make_entry(
        id=1,
        updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )]

    async def fake_fetch(*args, **kwargs):
        return entries

    monkeypatch.setattr("app.indexing.faq_indexer.fetch_faq", fake_fetch)
    mock_qdrant.get_indexed_at.return_value = "2026-06-04T10:00:00+00:00"

    stats = await index_faq(mock_nocodb, mock_qdrant, mock_embedder, force=False)
    assert stats.indexed == 0
    assert stats.skipped == 1
    mock_qdrant.upsert.assert_not_called()


async def test_index_faq_reindexes_updated(mock_nocodb, mock_qdrant, mock_embedder, monkeypatch):
    """Если UpdatedAt > indexed_at — переиндексируем."""
    entries = [_make_entry(
        id=1,
        updated_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )]

    async def fake_fetch(*args, **kwargs):
        return entries

    monkeypatch.setattr("app.indexing.faq_indexer.fetch_faq", fake_fetch)
    mock_qdrant.get_indexed_at.return_value = "2026-06-04T10:00:00+00:00"

    stats = await index_faq(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 1
    assert stats.skipped == 0
    mock_qdrant.delete_by_source.assert_called_once()
    mock_qdrant.upsert.assert_called_once()


async def test_index_faq_force_reindexes_all(mock_nocodb, mock_qdrant, mock_embedder, monkeypatch):
    """force=True переиндексирует, даже если UpdatedAt < indexed_at."""
    entries = [_make_entry(
        id=1,
        updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )]

    async def fake_fetch(*args, **kwargs):
        return entries

    monkeypatch.setattr("app.indexing.faq_indexer.fetch_faq", fake_fetch)
    mock_qdrant.get_indexed_at.return_value = "2026-06-04T10:00:00+00:00"

    stats = await index_faq(mock_nocodb, mock_qdrant, mock_embedder, force=True)
    assert stats.indexed == 1
    assert stats.skipped == 0


async def test_index_faq_continues_on_error(mock_nocodb, mock_qdrant, mock_embedder, monkeypatch):
    """Ошибка одной записи не ломает индексацию остальных."""
    entries = [
        _make_entry(id=1, question="ok 1"),
        _make_entry(id=2, question="fail"),
        _make_entry(id=3, question="ok 3"),
    ]

    async def fake_fetch(*args, **kwargs):
        return entries

    monkeypatch.setattr("app.indexing.faq_indexer.fetch_faq", fake_fetch)
    mock_qdrant.get_indexed_at.return_value = None

    # Эмбеддер падает на втором вызове
    call_count = {"n": 0}
    async def fake_embed(texts):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("embedder failure")
        return [[0.1] * 1024]

    mock_embedder.embed_documents = fake_embed

    stats = await index_faq(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 2
    assert stats.errors == 1