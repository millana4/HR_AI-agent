"""
Репозиторий документов из таблицы AI_DOCUMENTS в NocoDB.

Документы лежат как файлы в Selectel CDN, а в NocoDB — их метаданные
(title, url, type, description). Индексатор обходит эту таблицу, скачивает
файлы по URL и записывает чанки в Qdrant.

Дата последней индексации хранится в Qdrant в payload чанков (поле indexed_at).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from app.core.config import Config
from app.core.logging import get_logger
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)


DocumentType = Literal["blank", "regulation", "support"]


@dataclass
class Document:
    """Запись из AI_DOCUMENTS."""

    id: int
    title: str
    url: str
    doc_type: DocumentType
    description: str
    updated_at: datetime | None


def _parse_datetime(value: str | None) -> datetime | None:
    """NocoDB отдаёт даты в ISO-формате с tz, например '2026-06-02 09:19:05+00:00'."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning(f"Failed to parse datetime: {value!r}")
        return None


async def fetch_documents(
    client: NocoDBClient,
    correlation_id: str = "-",
) -> list[Document]:
    """
    Получить все активные документы из AI_DOCUMENTS.

    Возвращает только записи с Active=true. Документы с неизвестным Type
    пропускаются с warning.
    """
    raw_records = await client.list_records(
        table_id=Config.AI_DOCUMENTS_TABLE_ID,
        correlation_id=correlation_id,
    )

    documents: list[Document] = []
    for rec in raw_records:
        if not rec.get("Active"):
            continue

        doc_type = rec.get("Type")
        if doc_type not in ("blank", "regulation", "support"):
            logger.warning(
                f"Skipping document {rec.get('Id')}: unknown type {doc_type!r}",
                extra={"correlation_id": correlation_id},
            )
            continue

        documents.append(
            Document(
                id=rec["Id"],
                title=rec.get("Title", "").strip(),
                url=rec.get("URL", "").strip(),
                doc_type=doc_type,
                description=(rec.get("Description") or "").strip(),
                updated_at=_parse_datetime(rec.get("UpdatedAt")),
            )
        )

    logger.info(
        f"Fetched {len(documents)} active documents from NocoDB",
        extra={"correlation_id": correlation_id},
    )
    return documents