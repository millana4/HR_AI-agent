"""
Чтение из своднjq таблицs PIVOT_TABLE_ID.
Используется для построения словаря в PII-парсере.
"""
from app.core.config import Config
from app.core.logging import get_logger
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)


async def fetch_pivot(
    client: NocoDBClient,
    correlation_id: str = "-",
) -> list[str]:
    """
    Получить данные сводной таблицы.
    """
    records = await client.list_records(
        table_id=Config.PIVOT_TABLE_ID,
        fields=["FIO"],
        correlation_id=correlation_id,
    )

    fios = [r.get("FIO") for r in records if r.get("FIO")]

    logger.info(
        f"Fetched {len(fios)} employees with FIO",
        extra={"correlation_id": correlation_id},
    )
    return fios