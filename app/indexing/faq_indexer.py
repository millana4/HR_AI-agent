"""
Индексатор FAQ из NocoDB в Qdrant.

Логика инкрементальная: каждая FAQ-запись индексируется заново только если
её UpdatedAt в NocoDB новее, чем indexed_at в Qdrant. Сравнение по datetime,
обе стороны — в UTC.

FAQ-запись = один чанк (короткий текст, делить смысла нет).

source_id у FAQ имеет формат "nocodb_id:<Id>" — синтетический идентификатор
по полю Id из NocoDB.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.logging import get_logger
from app.rag.embedder import Embedder
from app.rag.qdrant_store import QdrantChunk, QdrantStore
from app.repositories.faq import FaqEntry, fetch_faq
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)


@dataclass
class IndexerStats:
    """Статистика прогона индексатора."""

    indexed: int = 0   # сколько записей проиндексировано (создано или обновлено)
    skipped: int = 0   # сколько пропущено (актуальны)
    errors: int = 0    # сколько упало с ошибкой

    def __str__(self) -> str:
        return f"indexed={self.indexed}, skipped={self.skipped}, errors={self.errors}"


def _make_faq_source_id(nocodb_id: int) -> str:
    """Идентификатор FAQ-записи в Qdrant. Использует Id из NocoDB."""
    return f"nocodb_id:{nocodb_id}"


def _make_faq_text(entry: FaqEntry) -> str:
    """Текст для эмбеддинга. Только Question + Answer, без Hidden_data и Link."""
    return f"Вопрос: {entry.question}\nОтвет: {entry.answer}"


def _needs_reindex(
    updated_at: datetime | None,
    indexed_at_iso: str | None,
) -> bool:
    """
    Решить, нужно ли переиндексировать запись.

    True если:
    - записи нет в Qdrant (indexed_at_iso is None), или
    - UpdatedAt в NocoDB новее, чем indexed_at в Qdrant.

    Если updated_at в NocoDB неизвестен — на всякий случай переиндексируем.
    """
    if indexed_at_iso is None:
        return True
    if updated_at is None:
        return True
    try:
        indexed_at = datetime.fromisoformat(indexed_at_iso)
    except ValueError:
        logger.warning(f"Failed to parse indexed_at: {indexed_at_iso!r}")
        return True
    return updated_at > indexed_at


async def index_faq(
    nocodb_client: NocoDBClient,
    qdrant_store: QdrantStore,
    embedder: Embedder,
    force: bool = False,
    correlation_id: str = "-",
) -> IndexerStats:
    """
    Проиндексировать FAQ из NocoDB в Qdrant.

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
    entries = await fetch_faq(nocodb_client, correlation_id=correlation_id)

    if not entries:
        logger.info(
            "No active FAQ entries to index",
            extra={"correlation_id": correlation_id},
        )
        return stats

    for entry in entries:
        source_id = _make_faq_source_id(entry.id)

        try:
            if not force:
                indexed_at_iso = await qdrant_store.get_indexed_at(
                    source_id, correlation_id=correlation_id
                )
                if not _needs_reindex(entry.updated_at, indexed_at_iso):
                    stats.skipped += 1
                    logger.debug(
                        f"FAQ {entry.id} is up to date, skipping",
                        extra={"correlation_id": correlation_id},
                    )
                    continue

            text = _make_faq_text(entry)
            vectors = await embedder.embed_documents([text])

            # Удаляем старые чанки (если были) перед записью новых
            await qdrant_store.delete_by_source(
                source_id, correlation_id=correlation_id
            )

            chunk = QdrantChunk(
                vector=vectors[0],
                source_type="faq",
                source_id=source_id,
                title=entry.question,
                text=text,
                chunk_index=0,
                indexed_at=datetime.now(timezone.utc).isoformat(),
            )
            await qdrant_store.upsert([chunk], correlation_id=correlation_id)
            stats.indexed += 1
            logger.info(
                f"Indexed FAQ {entry.id}: {entry.question[:50]}",
                extra={"correlation_id": correlation_id},
            )
        except Exception as exc:
            stats.errors += 1
            logger.exception(
                f"Failed to index FAQ {entry.id}: {exc}",
                extra={"correlation_id": correlation_id},
            )

    logger.info(
        f"FAQ indexing complete: {stats}",
        extra={"correlation_id": correlation_id},
    )
    return stats