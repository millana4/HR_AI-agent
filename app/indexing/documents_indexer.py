"""
Индексатор документов из NocoDB+CDN в Qdrant.

Логика инкрементальная: документ переиндексируется только если
его UpdatedAt в NocoDB новее, чем indexed_at в Qdrant.

Поведение по типу документа:
- blank: содержимое файла не индексируется. Создаётся один чанк = title + description.
         Это нужно, чтобы агент мог найти бланк по семантике и вернуть его URL.
- regulation / support: скачиваем файл из CDN, извлекаем текст, чанкуем, эмбеддим.
- Если из файла извлечь текст не удалось (PDF-скан, пустой docx) — fallback на
  индексацию только метаданных, как для blank.

source_id у документов — это их URL в CDN.
"""
from datetime import datetime, timezone

from app.core.logging import get_logger
from app.indexing.faq_indexer import IndexerStats, _needs_reindex
from app.indexing.file_readers import download_and_extract
from app.rag.chunker import split_text
from app.rag.embedder import Embedder
from app.rag.qdrant_store import QdrantChunk, QdrantStore, SourceType
from app.repositories.documents import Document, fetch_documents
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)


def _make_metadata_text(doc: Document) -> str:
    """
    Текст-метаданные для документа: заголовок + описание.

    Используется для blank-документов и как fallback для regulation/support,
    из которых не удалось извлечь содержимое.
    """
    if doc.description:
        return f"{doc.title}. {doc.description}"
    return doc.title


async def _index_one_document(
    doc: Document,
    qdrant_store: QdrantStore,
    embedder: Embedder,
    correlation_id: str,
) -> None:
    """
    Проиндексировать один документ: удалить старые чанки, написать новые.

    Кидает исключение при любой ошибке — вызывающий код решает что делать.
    """
    source_id = doc.url
    now_iso = datetime.now(timezone.utc).isoformat()

    # Решаем что класть в Qdrant — содержимое или только метаданные
    if doc.doc_type == "blank":
        texts_to_embed = [_make_metadata_text(doc)]
    else:
        # regulation / support — пробуем скачать и распарсить
        try:
            content = await download_and_extract(
                doc.url, correlation_id=correlation_id
            )
        except Exception as exc:
            # Скачивание/парсинг упали — индексируем только метаданные
            logger.warning(
                f"Failed to extract content from {doc.url}: {exc}. "
                f"Falling back to metadata-only indexing.",
                extra={"correlation_id": correlation_id},
            )
            content = ""

        if not content.strip():
            # Пустой файл или скан без текстового слоя
            texts_to_embed = [_make_metadata_text(doc)]
            logger.info(
                f"Document {doc.id} has no extractable content, indexing metadata only",
                extra={"correlation_id": correlation_id},
            )
        else:
            chunks = split_text(content, correlation_id=correlation_id)
            texts_to_embed = [c.text for c in chunks]

    # Эмбеддим все тексты одним batch (быстрее, чем по одному)
    vectors = await embedder.embed_documents(texts_to_embed)

    # Удаляем старые чанки документа, если были
    await qdrant_store.delete_by_source(source_id, correlation_id=correlation_id)

    # Записываем новые чанки
    qdrant_chunks = [
        QdrantChunk(
            vector=vector,
            source_type=doc.doc_type,  # blank / regulation / support
            source_id=source_id,
            title=doc.title,
            text=text,
            chunk_index=i,
            indexed_at=now_iso,
        )
        for i, (text, vector) in enumerate(zip(texts_to_embed, vectors))
    ]
    await qdrant_store.upsert(qdrant_chunks, correlation_id=correlation_id)


async def index_documents(
    nocodb_client: NocoDBClient,
    qdrant_store: QdrantStore,
    embedder: Embedder,
    force: bool = False,
    correlation_id: str = "-",
) -> IndexerStats:
    """
    Проиндексировать все документы из NocoDB в Qdrant.

    Args:
        nocodb_client: клиент NocoDB
        qdrant_store: уже подключённое хранилище Qdrant
        embedder: эмбеддер
        force: если True — переиндексировать всё, игнорируя дату
        correlation_id: для логов

    Returns:
        IndexerStats со статистикой.
    """
    stats = IndexerStats()
    documents = await fetch_documents(nocodb_client, correlation_id=correlation_id)

    if not documents:
        logger.info(
            "No active documents to index",
            extra={"correlation_id": correlation_id},
        )
        return stats

    for doc in documents:
        try:
            if not force:
                indexed_at_iso = await qdrant_store.get_indexed_at(
                    doc.url, correlation_id=correlation_id
                )
                if not _needs_reindex(doc.updated_at, indexed_at_iso):
                    stats.skipped += 1
                    logger.debug(
                        f"Document {doc.id} ({doc.title}) is up to date, skipping",
                        extra={"correlation_id": correlation_id},
                    )
                    continue

            await _index_one_document(
                doc, qdrant_store, embedder, correlation_id=correlation_id
            )
            stats.indexed += 1
            logger.info(
                f"Indexed document {doc.id}: {doc.title} (type={doc.doc_type})",
                extra={"correlation_id": correlation_id},
            )
        except Exception as exc:
            stats.errors += 1
            logger.exception(
                f"Failed to index document {doc.id} ({doc.title}): {exc}",
                extra={"correlation_id": correlation_id},
            )

    logger.info(
        f"Documents indexing complete: {stats}",
        extra={"correlation_id": correlation_id},
    )
    return stats