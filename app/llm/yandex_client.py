"""
Реализация LLM-клиента для Yandex AI Studio.

Использует OpenAI-совместимый Chat Completions API:
    POST https://llm.api.cloud.yandex.net/v1/chat/completions

Особенности:
- Аутентификация: Api-Key (ключ сервисного аккаунта) + folder_id.
- Модель указывается как URI gpt://<folder_id>/<model> в поле "model".
  Имя модели передаётся параметром в chat() — Pass 1 (lite) и Pass 2 (deepseek)
  используют разные модели через один клиент.
- Заголовок x-data-logging-enabled: false отключает сохранение запросов
  на серверах Яндекса (обязательно для ПД).
- Формат tools/tool_calls — современный OpenAI (не legacy functions как GigaChat).
  arguments приходит строкой JSON, парсим в dict.
"""
import json
from typing import Any

import httpx

from app.core.config import Config
from app.core.exceptions import LLMError, LLMTimeoutError, LLMAuthError
from app.core.logging import get_logger
from app.llm.base import BaseLLMClient, LLMResponse, Message, ToolCall, ToolSpec
from app.tools.registry import TOOLS, get_all_tool_names

logger = get_logger(__name__)


class YandexClient(BaseLLMClient):
    """LLM-клиент для Yandex AI Studio (OpenAI-совместимый API)."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=Config.LLM_TIMEOUT,
            trust_env=False,
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ============================================
    # chat
    # ============================================

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        correlation_id: str = "-",
        model: str | None = None,
    ) -> LLMResponse:
        """
        Отправить запрос в Yandex AI Studio.

        Args:
            model: короткое имя модели (напр. "yandexgpt-5-lite"). Если None —
                   берётся YANDEX_MODEL_PASS2 как дефолт.
        """
        model_name = model or Config.YANDEX_MODEL_PASS2
        model_uri = f"gpt://{Config.YANDEX_FOLDER_ID}/{model_name}"

        body = self._build_request_body(messages, tools, model_uri)

        logger.info(
            f"Yandex request: model={model_name}, messages={len(messages)}, "
            f"tools={len(tools) if tools else 0}",
            extra={"correlation_id": correlation_id},
        )

        headers = {
            "Authorization": f"Api-Key {Config.YANDEX_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Отключаем логирование запросов на стороне Яндекса (ПД!)
            "x-data-logging-enabled": "false",
            "x-folder-id": Config.YANDEX_FOLDER_ID,
        }

        url = f"{Config.YANDEX_API_URL.rstrip('/')}/chat/completions"

        try:
            response = await self._http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Yandex timeout ({Config.LLM_TIMEOUT}s) for model {model_name}"
            ) from exc
        except Exception as exc:
            raise LLMError(f"Yandex request failed: {exc}") from exc

        if response.status_code in (401, 403):
            # Невалидный ключ или нет средств — повод для алерта.
            raise LLMAuthError(
                f"Yandex auth/payment error: HTTP {response.status_code}, "
                f"body: {response.text[:300]}"
            )
        if response.status_code != 200:
            raise LLMError(
                f"Yandex error: HTTP {response.status_code}, "
                f"body: {response.text[:300]}"
            )

        return self._parse_response(response.json(), correlation_id)

    # ============================================
    # Внутренние методы
    # ============================================

    def _build_request_body(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None,
            model_uri: str,
    ) -> dict[str, Any]:
        """Сформировать тело запроса в OpenAI-совместимом формате."""
        body: dict[str, Any] = {
            "model": model_uri,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": Config.LLM_TEMPERATURE,
            "max_tokens": Config.LLM_MAX_TOKENS,
        }

        # DeepSeek — reasoning-модель: по умолчанию генерирует внутренние
        # рассуждения (reasoning_content), которые тратят токены и время, но
        # пользователю не нужны. Отключаем их. Для lite параметра быть не должно.
        if "deepseek" in model_uri.lower():
            body["reasoning_effort"] = "none"

        if tools:
            # Современный OpenAI-формат: tools + tool_choice
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            body["tool_choice"] = "auto"

        return body

    def _parse_response(
        self,
        body: dict[str, Any],
        correlation_id: str,
    ) -> LLMResponse:
        """Распарсить ответ Yandex (OpenAI-формат) в наш LLMResponse."""
        logger.debug(
            f"Yandex raw response: {body}",
            extra={"correlation_id": correlation_id},
        )
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Yandex response structure: {exc}") from exc

        # Tool calls (современный формат: message.tool_calls — список)
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw:
            tool_calls: list[ToolCall] = []
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                name = func.get("name", "")
                raw_args = func.get("arguments", "{}")
                # В OpenAI-формате arguments — строка JSON
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw_args or {}
                tool_calls.append(ToolCall(name=name, args=args))

            logger.info(
                f"Yandex tool calls: {[tc.name for tc in tool_calls]}",
                extra={"correlation_id": correlation_id},
            )
            return LLMResponse(type="tool_calls", tool_calls=tool_calls)

        # Текстовый ответ.
        content = message.get("content", "") or ""

        # Подстраховка для слабых моделей (yandexgpt-5-lite): иногда модель
        # пишет вызов инструмента ТЕКСТОМ в content вместо поля tool_calls,
        # например: 'search_internal\n{"query":"..."}' или просто 'search_internal'.
        # Пытаемся распознать это как tool_call.
        salvaged = self._salvage_tool_from_text(content)
        if salvaged is not None:
            logger.info(
                f"Yandex tool call (восстановлен из текста): {salvaged.name}",
                extra={"correlation_id": correlation_id},
            )
            return LLMResponse(type="tool_calls", tool_calls=[salvaged])

        logger.info(
            f"Yandex text response: {len(content)} chars",
            extra={"correlation_id": correlation_id},
        )
        return LLMResponse(type="text", content=content)


    def _salvage_tool_from_text(self, content: str) -> ToolCall | None:
        """
        Попытаться извлечь tool_call из текстового content.

        Слабые модели (lite) иногда пишут вызов инструмента текстом:
            'search_internal\n{"query":"бланк"}'
            'search_internal'
            'search_internal {"query":"бланк"}'
        Возвращает ToolCall, если первое слово — известный инструмент, иначе None.
        """
        if not content:
            return None
        text = content.strip()

        # Первое «слово» (до пробела/новой строки/фигурной скобки) — кандидат в имя.
        # Отделяем имя от возможного JSON с аргументами.
        import re
        match = re.match(r"^([a-z_]+)\b", text)
        if not match:
            return None
        name = match.group(1)
        if name not in get_all_tool_names():
            return None

        # Пытаемся найти JSON-аргументы в остатке строки.
        args: dict = {}
        rest = text[len(name):].strip()
        if rest:
            json_match = re.search(r"\{.*\}", rest, re.DOTALL)
            if json_match:
                try:
                    args = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    args = {}
        return ToolCall(name=name, args=args)

