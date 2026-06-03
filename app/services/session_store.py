"""
Хранилище истории разговора пользователя с агентом.

Ключ сессии: f"session:{user_id}:{YYYY-MM-DD}"
TTL: 24 часа, сдвигается при каждой записи.
Максимум: SESSION_MAX_MESSAGES пар «вопрос-ответ» (хранится 2 * N сообщений).

Структура в Redis:
    session:{user_id}:{date} → list of JSON-encoded messages
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
"""
import json
from datetime import date, timedelta
from typing import Literal

import redis.asyncio as aioredis

from app.core.config import Config
from app.core.exceptions import RepositoryError
from app.core.logging import get_logger


logger = get_logger(__name__)

Role = Literal["user", "assistant"]


def make_session_key(user_id: int, target_date: date | None = None) -> str:
    """Формирует ключ сессии: session:{user_id}:{YYYY-MM-DD}."""
    if target_date is None:
        target_date = date.today()
    return f"session:{user_id}:{target_date.isoformat()}"


class SessionStore:
    """
    Хранилище сессий через Redis.

    Использование:
        store = SessionStore()
        await store.connect()
        await store.append(user_id=123, role="user", content="Привет")
        history = await store.get_history(user_id=123)
        await store.disconnect()
    """

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None
        self._ttl_seconds = Config.SESSION_TTL_HOURS * 3600
        # SESSION_MAX_MESSAGES — это пар «вопрос-ответ», в Redis храним по два сообщения на пару
        self._max_messages = Config.SESSION_MAX_MESSAGES * 2

    async def connect(self) -> None:
        """Установить соединение с Redis."""
        self._client = aioredis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD or None,
            decode_responses=True,
        )
        try:
            await self._client.ping()
            logger.info(
                f"Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}"
            )
        except Exception as exc:
            raise RepositoryError(f"Failed to connect to Redis: {exc}") from exc

    async def disconnect(self) -> None:
        """Закрыть соединение с Redis."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RepositoryError("SessionStore is not connected. Call connect() first.")
        return self._client

    async def append(
        self,
        user_id: int,
        role: Role,
        content: str,
        target_date: date | None = None,
        correlation_id: str = "-",
    ) -> None:
        """
        Добавить сообщение в историю сессии.

        Старые сообщения автоматически вытесняются, если в истории больше max_messages.
        TTL обновляется при каждой записи.
        """
        client = self._get_client()
        key = make_session_key(user_id, target_date)
        message = json.dumps({"role": role, "content": content}, ensure_ascii=False)

        try:
            pipe = client.pipeline()
            pipe.rpush(key, message)
            pipe.ltrim(key, -self._max_messages, -1)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()
        except Exception as exc:
            logger.error(
                f"Failed to append to session {key}: {exc}",
                extra={"correlation_id": correlation_id},
            )
            raise RepositoryError(f"Session append failed: {exc}") from exc

        logger.debug(
            f"Appended {role} message to {key}",
            extra={"correlation_id": correlation_id},
        )

    async def get_history(
        self,
        user_id: int,
        target_date: date | None = None,
        correlation_id: str = "-",
    ) -> list[dict]:
        """
        Получить историю сессии.

        Возвращает список словарей вида {"role": "...", "content": "..."}.
        Если истории нет — пустой список.
        """
        client = self._get_client()
        key = make_session_key(user_id, target_date)

        try:
            raw_messages = await client.lrange(key, 0, -1)
        except Exception as exc:
            logger.error(
                f"Failed to get history {key}: {exc}",
                extra={"correlation_id": correlation_id},
            )
            raise RepositoryError(f"Session get failed: {exc}") from exc

        history = [json.loads(m) for m in raw_messages]
        logger.debug(
            f"Loaded {len(history)} messages from {key}",
            extra={"correlation_id": correlation_id},
        )
        return history

    async def clear(
        self,
        user_id: int,
        target_date: date | None = None,
        correlation_id: str = "-",
    ) -> None:
        """Удалить историю сессии."""
        client = self._get_client()
        key = make_session_key(user_id, target_date)

        try:
            await client.delete(key)
        except Exception as exc:
            raise RepositoryError(f"Session clear failed: {exc}") from exc

        logger.debug(
            f"Cleared session {key}",
            extra={"correlation_id": correlation_id},
        )