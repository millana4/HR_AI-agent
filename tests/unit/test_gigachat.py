"""Unit-тесты GigaChatClient с моком HTTP-запросов."""
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.core.exceptions import LLMError, LLMTimeoutError
from app.llm.base import Message, ToolSpec
from app.llm.gigachat import GigaChatClient, LLMQuotaExhaustedError


@pytest.fixture
def client():
    c = GigaChatClient()
    # Подсовываем "уже полученный" токен, чтобы не мокать ещё и auth
    c._access_token = "fake_token"
    c._token_expires_at = time.time() + 1800
    return c


def _make_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    """Создаёт mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    resp.text = text
    return resp


async def test_chat_text_response(client: GigaChatClient):
    """LLM ответила текстом."""
    fake_response = _make_response(
        200,
        {
            "choices": [
                {"message": {"role": "assistant", "content": "Привет!"}}
            ]
        },
    )
    client._http.post = AsyncMock(return_value=fake_response)

    result = await client.chat(
        messages=[Message(role="user", content="Привет")],
    )

    assert result.type == "text"
    assert result.content == "Привет!"
    assert result.tool_calls == []


async def test_chat_tool_call_response(client: GigaChatClient):
    """LLM вернула function_call."""
    fake_response = _make_response(
        200,
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "function_call": {
                            "name": "search_faq",
                            "arguments": {"query": "отпуск"},
                        },
                    }
                }
            ]
        },
    )
    client._http.post = AsyncMock(return_value=fake_response)

    tools = [
        ToolSpec(
            name="search_faq",
            description="Поиск",
            parameters={"type": "object", "properties": {}},
        )
    ]
    result = await client.chat(
        messages=[Message(role="user", content="Сколько дней отпуска?")],
        tools=tools,
    )

    assert result.type == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_faq"
    assert result.tool_calls[0].args == {"query": "отпуск"}


async def test_chat_fallback_through_models(client: GigaChatClient):
    """Если Max исчерпан — пробует Pro. Если Pro исчерпан — пробует Lite."""
    # Max → 402, Pro → 402, Lite → 200
    responses = [
        _make_response(402),
        _make_response(402),
        _make_response(
            200,
            {"choices": [{"message": {"content": "Ответ от Lite"}}]},
        ),
    ]
    client._http.post = AsyncMock(side_effect=responses)

    result = await client.chat(
        messages=[Message(role="user", content="Тест")],
    )

    assert result.type == "text"
    assert result.content == "Ответ от Lite"
    assert client._http.post.call_count == 3


async def test_chat_all_quotas_exhausted(client: GigaChatClient):
    """Все три модели исчерпаны → LLMQuotaExhaustedError."""
    responses = [_make_response(402), _make_response(402), _make_response(402)]
    client._http.post = AsyncMock(side_effect=responses)

    with pytest.raises(LLMQuotaExhaustedError):
        await client.chat(messages=[Message(role="user", content="Тест")])


async def test_chat_timeout(client: GigaChatClient):
    """httpx.TimeoutException превращается в LLMTimeoutError."""
    client._http.post = AsyncMock(
        side_effect=httpx.TimeoutException("timeout")
    )

    with pytest.raises(LLMTimeoutError):
        await client.chat(messages=[Message(role="user", content="Тест")])


async def test_chat_5xx_error(client: GigaChatClient):
    """5xx-ошибка превращается в LLMError."""
    client._http.post = AsyncMock(return_value=_make_response(500, text="Server error"))

    with pytest.raises(LLMError):
        await client.chat(messages=[Message(role="user", content="Тест")])


async def test_chat_malformed_response(client: GigaChatClient):
    """Невалидная структура ответа → LLMError."""
    client._http.post = AsyncMock(
        return_value=_make_response(200, {"unexpected": "format"})
    )

    with pytest.raises(LLMError):
        await client.chat(messages=[Message(role="user", content="Тест")])