"""
Юнит-тесты YandexClient с моком HTTP. По минимуму — ключевая логика:
  - reasoning_effort=none для deepseek (и отсутствие для lite)
  - salvage tool-call из текстового content (слабая lite-модель)
  - извлечение model/tokens из ответа (чистка gpt://-префикса)
  - 401/403 → LLMAuthError
"""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.core.exceptions import LLMAuthError, LLMError
from app.llm.base import Message, ToolSpec
from app.llm.yandex_client import YandexClient


@pytest.fixture
def client():
    return YandexClient()


def _make_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    resp.text = text
    return resp


def _ok(message: dict, model: str = "gpt://b1gfolder/yandexgpt-5-lite",
        total_tokens: int = 42) -> dict:
    """Готовый OpenAI-совместимый ответ с одним choice."""
    return {
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"total_tokens": total_tokens},
    }


# ============================================
# reasoning_effort
# ============================================

async def test_deepseek_gets_reasoning_none(client: YandexClient):
    """Для deepseek в теле запроса проставляется reasoning_effort=none."""
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured["body"] = json
        return _make_response(200, _ok({"role": "assistant", "content": "ок"}))

    client._http.post = AsyncMock(side_effect=fake_post)

    await client.chat(
        messages=[Message(role="user", content="привет")],
        model="deepseek-v4-flash",
    )
    assert captured["body"].get("reasoning_effort") == "none"


async def test_lite_has_no_reasoning_param(client: YandexClient):
    """Для lite параметра reasoning_effort быть не должно."""
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured["body"] = json
        return _make_response(200, _ok({"role": "assistant", "content": "ок"}))

    client._http.post = AsyncMock(side_effect=fake_post)

    await client.chat(
        messages=[Message(role="user", content="привет")],
        model="yandexgpt-5-lite",
    )
    assert "reasoning_effort" not in captured["body"]


# ============================================
# salvage tool из текста
# ============================================

async def test_salvages_tool_from_text(client: YandexClient):
    """Если lite написала имя инструмента текстом — восстанавливаем tool_call."""
    body = _ok({
        "role": "assistant",
        "content": 'search_internal\n{"query":"бланк отпуска"}',
    })
    client._http.post = AsyncMock(return_value=_make_response(200, body))

    result = await client.chat(
        messages=[Message(role="user", content="найди бланк")],
        model="yandexgpt-5-lite",
    )
    assert result.type == "tool_calls"
    assert result.tool_calls[0].name == "search_internal"
    assert result.tool_calls[0].args == {"query": "бланк отпуска"}


async def test_plain_text_stays_text(client: YandexClient):
    """Обычный текст (не имя инструмента) остаётся текстовым ответом."""
    body = _ok({"role": "assistant", "content": "Просто ответ пользователю."})
    client._http.post = AsyncMock(return_value=_make_response(200, body))

    result = await client.chat(
        messages=[Message(role="user", content="привет")],
        model="deepseek-v4-flash",
    )
    assert result.type == "text"
    assert result.content == "Просто ответ пользователю."


# ============================================
# model / tokens
# ============================================

async def test_extracts_model_and_tokens(client: YandexClient):
    """Имя модели чистится от gpt://folder/, tokens берётся из usage."""
    body = _ok(
        {"role": "assistant", "content": "ок"},
        model="gpt://b1gfolder/yandexgpt-5-lite",
        total_tokens=123,
    )
    client._http.post = AsyncMock(return_value=_make_response(200, body))

    result = await client.chat(
        messages=[Message(role="user", content="привет")],
        model="yandexgpt-5-lite",
    )
    assert result.model == "yandexgpt-5-lite"
    assert result.tokens == 123


async def test_cleans_deepseek_latest_suffix(client: YandexClient):
    """gpt://deepseek-v4-flash/latest → deepseek-v4-flash (отбрасываем latest)."""
    body = _ok(
        {"role": "assistant", "content": "ок"},
        model="gpt://deepseek-v4-flash/latest",
    )
    client._http.post = AsyncMock(return_value=_make_response(200, body))

    result = await client.chat(
        messages=[Message(role="user", content="привет")],
        model="deepseek-v4-flash",
    )
    assert result.model == "deepseek-v4-flash"


# ============================================
# Ошибки авторизации/оплаты
# ============================================

@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_raises_llm_auth_error(client: YandexClient, status: int):
    """401/403 → LLMAuthError (повод для алерта)."""
    client._http.post = AsyncMock(
        return_value=_make_response(status, text="forbidden")
    )
    with pytest.raises(LLMAuthError):
        await client.chat(
            messages=[Message(role="user", content="привет")],
            model="yandexgpt-5-lite",
        )


async def test_5xx_raises_llm_error(client: YandexClient):
    """5xx → обычный LLMError."""
    client._http.post = AsyncMock(
        return_value=_make_response(500, text="server error")
    )
    with pytest.raises(LLMError):
        await client.chat(
            messages=[Message(role="user", content="привет")],
            model="yandexgpt-5-lite",
        )