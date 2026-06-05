"""Тесты индексатора документов."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.indexing.documents_indexer import (
    _index_one_document,
    _make_metadata_text,
    index_documents,
)
from app.repositories.documents import Document


def _make_doc(
    id: int = 1,
    title: str = "Doc",
    url: str = "https://cdn.example.com/doc.pdf",
    doc_type: str = "support",
    description: str = "",
    updated_at: datetime | None = None,
) -> Document:
    return Document(
        id=id, title=title, url=url, doc_type=doc_type,
        description=description,
        updated_at=updated_at or datetime(2026, 6, 4, 10, 0, 0, tzinfo=timezone.utc),
    )


# --- _make_metadata_text ---

def test_make_metadata_text_with_description():
    doc = _make_doc(title="Welcomebook", description="Буклет для сотрудников")
    text = _make_metadata_text(doc)
    assert text == "Welcomebook. Буклет для сотрудников"


def test_make_metadata_text_without_description():
    doc = _make_doc(title="Welcomebook", description="")
    text = _make_metadata_text(doc)
    assert text == "Welcomebook"


# --- _index_one_document ---

@pytest.fixture
def mock_qdrant():
    return AsyncMock()


@pytest.fixture
def mock_embedder():
    m = AsyncMock()
    m.embed_documents.return_value = [[0.1] * 1024]
    return m


async def test_index_one_document_blank_uses_metadata_only(
    mock_qdrant, mock_embedder, monkeypatch
):
    """Для blank содержимое файла НЕ скачивается, индексируется только title+description."""
    download_called = {"n": 0}

    async def fake_download(url, correlation_id="-"):
        download_called["n"] += 1
        return "this should not be used"

    monkeypatch.setattr(
        "app.indexing.documents_indexer.download_and_extract", fake_download
    )

    doc = _make_doc(
        title="Заявление на отпуск",
        description="Заявление для сотрудников",
        doc_type="blank",
    )
    await _index_one_document(doc, mock_qdrant, mock_embedder, correlation_id="-")

    # Скачивание не вызывалось
    assert download_called["n"] == 0
    # Эмбеддер получил один текст
    mock_embedder.embed_documents.assert_called_once()
    call_arg = mock_embedder.embed_documents.call_args.args[0]
    assert len(call_arg) == 1
    assert "Заявление" in call_arg[0]
    # Чанк записан с source_type=blank
    mock_qdrant.upsert.assert_called_once()
    chunks_arg = mock_qdrant.upsert.call_args.args[0]
    assert chunks_arg[0].source_type == "blank"


async def test_index_one_document_support_extracts_content(
    mock_qdrant, mock_embedder, monkeypatch
):
    """Для support скачиваем файл, чанкуем содержимое."""
    async def fake_download(url, correlation_id="-"):
        return "Long text content. " * 200  # ~600 слов

    monkeypatch.setattr(
        "app.indexing.documents_indexer.download_and_extract", fake_download
    )

    # embed_documents должен вернуть столько векторов, сколько чанков
    mock_embedder.embed_documents.side_effect = lambda texts: [[0.1] * 1024] * len(texts)

    doc = _make_doc(title="Welcomebook", doc_type="support")
    await _index_one_document(doc, mock_qdrant, mock_embedder, correlation_id="-")

    # Эмбеддер получил несколько чанков (текст длинный)
    mock_embedder.embed_documents.assert_called_once()
    chunks_arg = mock_qdrant.upsert.call_args.args[0]
    assert len(chunks_arg) >= 1
    assert chunks_arg[0].source_type == "support"


async def test_index_one_document_regulation_extracts_content(
    mock_qdrant, mock_embedder, monkeypatch
):
    """Для regulation также скачиваем содержимое."""
    async def fake_download(url, correlation_id="-"):
        return "Short regulation text."

    monkeypatch.setattr(
        "app.indexing.documents_indexer.download_and_extract", fake_download
    )

    doc = _make_doc(title="Reg 1", doc_type="regulation")
    await _index_one_document(doc, mock_qdrant, mock_embedder, correlation_id="-")

    chunks_arg = mock_qdrant.upsert.call_args.args[0]
    assert chunks_arg[0].source_type == "regulation"


async def test_index_one_document_fallback_on_empty_content(
    mock_qdrant, mock_embedder, monkeypatch
):
    """Если файл пустой (например, PDF-скан) — fallback на метаданные."""
    async def fake_download(url, correlation_id="-"):
        return ""

    monkeypatch.setattr(
        "app.indexing.documents_indexer.download_and_extract", fake_download
    )

    doc = _make_doc(
        title="Empty Doc",
        description="Description here",
        doc_type="support",
    )
    await _index_one_document(doc, mock_qdrant, mock_embedder, correlation_id="-")

    # Эмбеддер вызван с одним текстом (метаданные)
    call_arg = mock_embedder.embed_documents.call_args.args[0]
    assert len(call_arg) == 1
    assert "Empty Doc" in call_arg[0]


async def test_index_one_document_fallback_on_download_error(
    mock_qdrant, mock_embedder, monkeypatch
):
    """Если скачивание упало — fallback на метаданные, не пробрасываем."""
    async def fake_download(url, correlation_id="-"):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        "app.indexing.documents_indexer.download_and_extract", fake_download
    )

    doc = _make_doc(title="Doc", description="desc", doc_type="support")
    # Не должно упасть
    await _index_one_document(doc, mock_qdrant, mock_embedder, correlation_id="-")

    # Зато проиндексировались метаданные
    call_arg = mock_embedder.embed_documents.call_args.args[0]
    assert len(call_arg) == 1


async def test_index_one_document_deletes_old_chunks_first(
    mock_qdrant, mock_embedder, monkeypatch
):
    """Перед записью новых чанков старые удаляются."""
    async def fake_download(url, correlation_id="-"):
        return "content"

    monkeypatch.setattr(
        "app.indexing.documents_indexer.download_and_extract", fake_download
    )

    doc = _make_doc(url="https://cdn.example.com/x.pdf", doc_type="support")
    await _index_one_document(doc, mock_qdrant, mock_embedder, correlation_id="-")

    mock_qdrant.delete_by_source.assert_called_once_with(
        "https://cdn.example.com/x.pdf", correlation_id="-"
    )


# --- index_documents (full flow) ---

@pytest.fixture
def mock_nocodb():
    return AsyncMock()


async def test_index_documents_empty_list(
    mock_nocodb, mock_qdrant, mock_embedder, monkeypatch
):
    """Если документов нет — статистика нулевая."""
    async def fake_fetch(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "app.indexing.documents_indexer.fetch_documents", fake_fetch
    )

    stats = await index_documents(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 0
    assert stats.skipped == 0


async def test_index_documents_indexes_new(
    mock_nocodb, mock_qdrant, mock_embedder, monkeypatch
):
    """Документы, которых нет в Qdrant, индексируются."""
    async def fake_fetch(*args, **kwargs):
        return [_make_doc(id=1, doc_type="blank")]

    monkeypatch.setattr(
        "app.indexing.documents_indexer.fetch_documents", fake_fetch
    )
    mock_qdrant.get_indexed_at.return_value = None

    stats = await index_documents(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 1
    assert stats.skipped == 0


async def test_index_documents_skips_unchanged(
    mock_nocodb, mock_qdrant, mock_embedder, monkeypatch
):
    """Если UpdatedAt < indexed_at — пропускаем."""
    async def fake_fetch(*args, **kwargs):
        return [_make_doc(
            id=1,
            doc_type="blank",
            updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )]

    monkeypatch.setattr(
        "app.indexing.documents_indexer.fetch_documents", fake_fetch
    )
    mock_qdrant.get_indexed_at.return_value = "2026-06-04T10:00:00+00:00"

    stats = await index_documents(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 0
    assert stats.skipped == 1


async def test_index_documents_reindexes_updated(
    mock_nocodb, mock_qdrant, mock_embedder, monkeypatch
):
    """Если UpdatedAt > indexed_at — переиндексируем."""
    async def fake_fetch(*args, **kwargs):
        return [_make_doc(
            id=1,
            doc_type="blank",
            updated_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
        )]

    monkeypatch.setattr(
        "app.indexing.documents_indexer.fetch_documents", fake_fetch
    )
    mock_qdrant.get_indexed_at.return_value = "2026-06-04T10:00:00+00:00"

    stats = await index_documents(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 1


async def test_index_documents_force(
    mock_nocodb, mock_qdrant, mock_embedder, monkeypatch
):
    """force=True переиндексирует даже актуальное."""
    async def fake_fetch(*args, **kwargs):
        return [_make_doc(
            id=1,
            doc_type="blank",
            updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )]

    monkeypatch.setattr(
        "app.indexing.documents_indexer.fetch_documents", fake_fetch
    )
    mock_qdrant.get_indexed_at.return_value = "2026-06-04T10:00:00+00:00"

    stats = await index_documents(
        mock_nocodb, mock_qdrant, mock_embedder, force=True
    )
    assert stats.indexed == 1


async def test_index_documents_continues_on_error(
    mock_nocodb, mock_qdrant, mock_embedder, monkeypatch
):
    """Ошибка одного документа не ломает остальные."""
    docs = [
        _make_doc(id=1, doc_type="blank", title="ok 1"),
        _make_doc(id=2, doc_type="blank", title="fail"),
        _make_doc(id=3, doc_type="blank", title="ok 3"),
    ]

    async def fake_fetch(*args, **kwargs):
        return docs

    monkeypatch.setattr(
        "app.indexing.documents_indexer.fetch_documents", fake_fetch
    )
    mock_qdrant.get_indexed_at.return_value = None

    call_count = {"n": 0}
    async def fake_embed(texts):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("embedder failure")
        return [[0.1] * 1024] * len(texts)

    mock_embedder.embed_documents = fake_embed

    stats = await index_documents(mock_nocodb, mock_qdrant, mock_embedder)
    assert stats.indexed == 2
    assert stats.errors == 1