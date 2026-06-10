"""
HTTP-клиент к NocoDB API.

Используется агентом для:
- Чтения сводной таблицы
- Чтения FAQ (search_faq tool, шаг 13)
- Чтения списка документов (для индексации, шаг 11)
- Записи аналитики (шаг 14)
"""
from typing import Any

import httpx

from app.core.config import Config
from app.core.exceptions import RepositoryError
from app.core.logging import get_logger


logger = get_logger(__name__)


class NocoDBClient:
    """Тонкая обёртка над NocoDB REST API."""

    def __init__(self) -> None:
        self._base_url = Config.NOCODB_SERVER.rstrip("/")
        self._headers = {
            "xc-token": Config.NOCODB_API_TOKEN,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # trust_env=False — игнорируем системные прокси,
        # как в GigaChatClient
        self._http = httpx.AsyncClient(timeout=30, trust_env=False)

    async def close(self) -> None:
        await self._http.aclose()

    async def list_records(
        self,
        table_id: str,
        fields: list[str] | None = None,
        where: str | None = None,
        limit: int = 1000,
        correlation_id: str = "-",
    ) -> list[dict[str, Any]]:
        """
        Получить записи из таблицы.

        Args:
            table_id: ID таблицы в NocoDB
            fields: список полей для выборки (None — все)
            where: фильтр в формате NocoDB '(field,eq,value)'
            limit: максимум записей за один запрос
            correlation_id: для трассировки

        Returns:
            Список словарей-записей.

        Raises:
            RepositoryError при HTTP-ошибке или невалидном ответе.
        """
        url = f"{self._base_url}/api/v2/tables/{table_id}/records"
        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = ",".join(fields)
        if where:
            params["where"] = where

        all_records: list[dict[str, Any]] = []
        offset = 0

        while True:
            params["offset"] = offset
            try:
                response = await self._http.get(
                    url, headers=self._headers, params=params
                )
            except Exception as exc:
                raise RepositoryError(f"NocoDB request failed: {exc}") from exc

            if response.status_code != 200:
                raise RepositoryError(
                    f"NocoDB error: HTTP {response.status_code}, "
                    f"body: {response.text[:300]}"
                )

            try:
                body = response.json()
            except Exception as exc:
                raise RepositoryError(f"NocoDB returned invalid JSON: {exc}") from exc

            page = body.get("list", [])
            all_records.extend(page)

            page_info = body.get("pageInfo", {})
            if page_info.get("isLastPage", True):
                break
            offset += limit

        logger.debug(
            f"NocoDB list_records({table_id}): got {len(all_records)} records",
            extra={"correlation_id": correlation_id},
        )
        return all_records

    async def create_record(
            self,
            table_id: str,
            data: dict[str, Any],
            correlation_id: str = "-",
    ) -> dict[str, Any]:
        """
        Создать запись в таблице.

        Args:
            table_id: ID таблицы в NocoDB
            data: словарь полей записи
            correlation_id: для трассировки

        Returns:
            Созданная запись (ответ NocoDB).

        Raises:
            RepositoryError при HTTP-ошибке или невалидном ответе.
        """
        url = f"{self._base_url}/api/v2/tables/{table_id}/records"
        try:
            response = await self._http.post(
                url, headers=self._headers, json=data
            )
        except Exception as exc:
            raise RepositoryError(f"NocoDB create request failed: {exc}") from exc

        if response.status_code not in (200, 201):
            raise RepositoryError(
                f"NocoDB create error: HTTP {response.status_code}, "
                f"body: {response.text[:300]}"
            )

        try:
            body = response.json()
        except Exception as exc:
            raise RepositoryError(f"NocoDB returned invalid JSON: {exc}") from exc

        logger.debug(
            f"NocoDB create_record({table_id}): создана запись",
            extra={"correlation_id": correlation_id},
        )
        return body