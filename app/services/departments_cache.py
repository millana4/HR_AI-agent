"""
Кеш списков отделов компаний Мавис и Вотоня.

Списки отделов нужны в системном промпте, чтобы LLM выбирала отдел из
реального справочника, а не угадывала. Чтобы не ходить в NocoDB на каждый
запрос, список строится раз в сутки и кешируется в Redis с TTL 24 часа.

Источник — таблицы ATS_MAVIS_BOOK_ID и ATS_VOTONIA_BOOK_ID в NocoDB.
Из каждой берём уникальные непустые значения поля Department.

Ключи в Redis:
    departments:mavis
    departments:votonia
"""
import json

import redis.asyncio as aioredis

from app.core.config import Config
from app.core.logging import get_logger
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)


_TTL_SECONDS = 24 * 3600
_KEY_MAVIS = "departments:mavis"
_KEY_VOTONIA = "departments:votonia"


class DepartmentsCache:
    """
    Кеш списков отделов в Redis.

    Использование:
        cache = DepartmentsCache()
        await cache.connect()
        mavis, votonia = await cache.get_departments(nocodb_client, correlation_id)
        await cache.disconnect()
    """

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Установить соединение с Redis (тот же сервер, что и сессии)."""
        self._client = aioredis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD or None,
            decode_responses=True,
        )
        await self._client.ping()
        logger.info("DepartmentsCache connected to Redis")

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("DepartmentsCache is not connected. Call connect() first.")
        return self._client

    async def get_departments(
        self,
        nocodb_client: NocoDBClient,
        correlation_id: str = "-",
    ) -> tuple[list[str], list[str]]:
        """
        Получить списки отделов (Мавис, Вотоня).

        Если в Redis есть свежий кеш — берём из него. Иначе строим из NocoDB
        и кладём в Redis на сутки.
        """
        client = self._get_client()

        cached_mavis = await client.get(_KEY_MAVIS)
        cached_votonia = await client.get(_KEY_VOTONIA)

        if cached_mavis is not None and cached_votonia is not None:
            logger.debug(
                "Departments взяты из кеша Redis",
                extra={"correlation_id": correlation_id},
            )
            return json.loads(cached_mavis), json.loads(cached_votonia)

        logger.info(
            "Departments кеш пуст — строим из NocoDB",
            extra={"correlation_id": correlation_id},
        )
        mavis = await self._fetch_departments(
            nocodb_client, Config.ATS_MAVIS_BOOK_ID, "Мавис", correlation_id
        )
        votonia = await self._fetch_departments(
            nocodb_client, Config.ATS_VOTONIA_BOOK_ID, "Вотоня", correlation_id
        )

        # Кешируем на сутки.
        await client.set(_KEY_MAVIS, json.dumps(mavis, ensure_ascii=False), ex=_TTL_SECONDS)
        await client.set(_KEY_VOTONIA, json.dumps(votonia, ensure_ascii=False), ex=_TTL_SECONDS)

        return mavis, votonia

    async def _fetch_departments(
        self,
        nocodb_client: NocoDBClient,
        table_id: str | None,
        company: str,
        correlation_id: str,
    ) -> list[str]:
        """Получить уникальные отделы из одной таблицы справочника."""
        if not table_id:
            logger.warning(
                f"Не задан ID таблицы справочника {company} — список отделов пуст",
                extra={"correlation_id": correlation_id},
            )
            return []

        records = await nocodb_client.list_records(
            table_id=table_id,
            fields=["Department"],
            correlation_id=correlation_id,
        )
        # Уникальные непустые отделы, сохраняя стабильный порядок (сортировкой).
        departments = sorted({
            (r.get("Department") or "").strip()
            for r in records
            if (r.get("Department") or "").strip()
        })
        logger.info(
            f"Справочник {company}: {len(departments)} отделов",
            extra={"correlation_id": correlation_id},
        )
        return departments