"""
Репозиторий FAQ из таблицы AI_FAQ в NocoDB.

Каждая запись — пара "вопрос/ответ", опционально с ссылкой, attachment'ом
и скрытыми данными (Hidden_data). Векторизуем "Question + Answer" как один чанк,
остальные поля сохраняем в payload Qdrant для использования tool'ом.
"""
from dataclasses import dataclass
from datetime import datetime

from app.core.config import Config
from app.core.logging import get_logger
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)


@dataclass
class FaqEntry:
    """Запись из AI_FAQ."""

    id: int
    question: str
    answer: str
    link: str | None
    attachment: str | None
    hidden_data: str | None
    updated_at: datetime | None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning(f"Failed to parse datetime: {value!r}")
        return None


async def fetch_faq(
    client: NocoDBClient,
    correlation_id: str = "-",
) -> list[FaqEntry]:
    """
    Получить все активные записи FAQ.

    Возвращает только Active=true. Пустые Question/Answer пропускаются.
    """
    raw_records = await client.list_records(
        table_id=Config.AI_FAQ_TABLE_ID,
        correlation_id=correlation_id,
    )

    entries: list[FaqEntry] = []
    for rec in raw_records:
        if not rec.get("Active"):
            continue

        question = (rec.get("Question") or "").strip()
        answer = (rec.get("Answer") or "").strip()

        if not question or not answer:
            logger.warning(
                f"Skipping FAQ {rec.get('Id')}: empty question or answer",
                extra={"correlation_id": correlation_id},
            )
            continue

        entries.append(
            FaqEntry(
                id=rec["Id"],
                question=question,
                answer=answer,
                link=(rec.get("Link") or None) or None,
                attachment=(rec.get("Attachment") or None) or None,
                hidden_data=(rec.get("Hidden_data") or None) or None,
                updated_at=_parse_datetime(rec.get("UpdatedAt")),
            )
        )

    logger.info(
        f"Fetched {len(entries)} active FAQ entries from NocoDB",
        extra={"correlation_id": correlation_id},
    )
    return entries