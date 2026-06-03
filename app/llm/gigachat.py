"""
Реализация LLM-клиента для GigaChat (Сбер).

Особенности GigaChat:
- OAuth-авторизация (получаем access_token, который живёт 30 минут)
- В .env лежит Authorization key (base64 от client_id:client_secret)
- API формат OpenAI-совместимый
- Три модели: GigaChat-2-Max (мощная), GigaChat-2-Pro, GigaChat-2 (Lite, самая простая)

Стратегия моделей: используем сильную (Max), при исчерпании квоты — fallback на Pro, потом на Lite.
Когда все три исчерпаны — кидаем LLMQuotaExhaustedError.
"""
import json
import ssl
import time
import uuid
from typing import Any

import httpx

from app.core.config import Config
from app.core.exceptions import LLMError, LLMTimeoutError
from app.core.logging import get_logger
from app.llm.base import BaseLLMClient, LLMResponse, Message, ToolCall, ToolSpec


logger = get_logger(__name__)


# Приоритет моделей: сначала самая мощная, при квоте → fallback
MODELS_PRIORITY = ["GigaChat-2-Max", "GigaChat-2-Pro", "GigaChat-2"]


class LLMQuotaExhaustedError(LLMError):
    """Все доступные модели исчерпали лимит токенов."""


class GigaChatClient(BaseLLMClient):
    """LLM-клиент для GigaChat API."""

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        # GigaChat использует самоподписанные сертификаты Минцифры — verify=False
        # В продакшене лучше скачать и подложить сертификаты Минцифры
        self._http = httpx.AsyncClient(
            verify=False,
            timeout=Config.LLM_TIMEOUT,
            trust_env=False,
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ============================================
    # OAuth
    # ============================================

    async def _get_access_token(self, correlation_id: str) -> str:
        """Получить access_token. Кэшируется до истечения срока."""
        # Если токен ещё живёт минимум 30 секунд — переиспользуем
        if self._access_token and time.time() < self._token_expires_at - 30:
            return self._access_token

        headers = {
            "Authorization": f"Basic {Config.GIGACHAT_AUTH_KEY}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
        }
        data = {"scope": Config.GIGACHAT_SCOPE}

        try:
            response = await self._http.post(
                Config.GIGACHAT_AUTH_URL,
                headers=headers,
                data=data,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"GigaChat auth timeout: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"GigaChat auth request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"GigaChat auth failed: HTTP {response.status_code}, body: {response.text[:200]}"
            )

        body = response.json()
        self._access_token = body["access_token"]
        # GigaChat возвращает expires_at в миллисекундах
        self._token_expires_at = body["expires_at"] / 1000

        logger.info(
            "GigaChat access_token obtained",
            extra={"correlation_id": correlation_id},
        )
        return self._access_token

    # ============================================
    # chat
    # ============================================

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        correlation_id: str = "-",
    ) -> LLMResponse:
        """Отправить запрос в GigaChat. Пробует модели по приоритету при исчерпании квот."""
        access_token = await self._get_access_token(correlation_id)
        request_body_base = self._build_request_body(messages, tools)

        for model in MODELS_PRIORITY:
            body = {**request_body_base, "model": model}

            logger.info(
                f"GigaChat request: model={model}, messages={len(messages)}, "
                f"tools={len(tools) if tools else 0}",
                extra={"correlation_id": correlation_id},
            )

            try:
                response = await self._http.post(
                    Config.GIGACHAT_API_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-Request-ID": correlation_id,
                    },
                    json=body,
                )
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(
                    f"GigaChat timeout ({Config.LLM_TIMEOUT}s) for model {model}"
                ) from exc
            except Exception as exc:
                raise LLMError(f"GigaChat request failed: {exc}") from exc

            # Проверяем, не исчерпан ли лимит
            if self._is_quota_exhausted(response):
                logger.warning(
                    f"GigaChat quota exhausted for model {model}, trying next",
                    extra={"correlation_id": correlation_id},
                )
                continue

            if response.status_code != 200:
                raise LLMError(
                    f"GigaChat error: HTTP {response.status_code}, "
                    f"body: {response.text[:300]}"
                )

            return self._parse_response(response.json(), correlation_id)

        # Все модели исчерпали лимит
        raise LLMQuotaExhaustedError(
            "All GigaChat models exhausted their token quota"
        )

    # ============================================
    # Внутренние методы
    # ============================================

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
    ) -> dict[str, Any]:
        """Сформировать тело запроса. model подставляется отдельно в chat()."""
        body: dict[str, Any] = {
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": Config.LLM_TEMPERATURE,
            "max_tokens": Config.LLM_MAX_TOKENS,
        }

        if tools:
            body["functions"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
            body["function_call"] = "auto"

        return body

    def _is_quota_exhausted(self, response: httpx.Response) -> bool:
        """Определить, исчерпан ли лимит токенов."""
        # GigaChat при исчерпании квоты возвращает 402 Payment Required
        # или 429 с конкретным сообщением
        if response.status_code == 402:
            return True
        if response.status_code == 429:
            return True
        return False

    def _parse_response(
        self,
        body: dict[str, Any],
        correlation_id: str,
    ) -> LLMResponse:
        """Распарсить ответ GigaChat в наш LLMResponse."""
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected GigaChat response structure: {exc}") from exc

        # Function call (LLM вызывает tool)
        function_call = message.get("function_call")
        if function_call:
            name = function_call.get("name", "")
            raw_args = function_call.get("arguments", {})
            # GigaChat возвращает arguments как dict (не строку как у OpenAI)
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args

            logger.info(
                f"GigaChat tool call: {name}",
                extra={"correlation_id": correlation_id},
            )
            return LLMResponse(
                type="tool_calls",
                tool_calls=[ToolCall(name=name, args=args)],
            )

        # Текстовый ответ
        content = message.get("content", "")
        logger.info(
            f"GigaChat text response: {len(content)} chars",
            extra={"correlation_id": correlation_id},
        )
        return LLMResponse(type="text", content=content)