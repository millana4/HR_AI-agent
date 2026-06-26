"""
Интеграционный тест Yandex-петли (process_request_yandex).

По минимуму — главные пути:
  - Pass 1 (lite) → search_internal → Pass 2 (deepseek) с подстановкой hidden_data
  - Pass 1 → bot_command (возврат боту, без Pass 2)
  - Pass 1 → answer_general → Pass 2 general
  - Pass 1 (lite) вернул текст → эскалация на сильную модель

Особенность Yandex-петли: Pass 1 — ЧИСТЫЙ КЛАССИФИКАТОР, всегда tool-call.
chat() принимает параметр model (lite на Pass 1, deepseek на Pass 2).
Моки всех внешних зависимостей; FastAPI/HTTP не дёргаем.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.schemas import TextResponse, ToolCallResponse
from app.llm.base import LLMResponse, ToolCall as LLMToolCall
from app.rag.qdrant_store import SearchResult
from app.services.agent_common import AgentLoop
from app.services.agent_loop_yandex import process_request_yandex
from app.services.pii_parser import PiiParseResult


def _pii(masked_text: str, found_names: list[str]) -> PiiParseResult:
    return PiiParseResult(masked_text=masked_text, found_names=found_names)


def _resp(**kwargs) -> LLMResponse:
    """LLMResponse с дефолтными model/tokens (чтобы аналитика не падала)."""
    kwargs.setdefault("model", "yandexgpt-5-lite")
    kwargs.setdefault("tokens", 10)
    return LLMResponse(**kwargs)


@pytest.fixture
def agent():
    llm = AsyncMock()

    session_store = AsyncMock()
    session_store.get_history = AsyncMock(return_value=[])
    session_store.append = AsyncMock()

    pii_parser = MagicMock()
    pii_parser.parse = MagicMock(
        side_effect=lambda text, correlation_id="-": _pii(text, [])
    )

    nocodb_client = MagicMock()

    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)

    qdrant_store = AsyncMock()
    qdrant_store.search = AsyncMock(
        return_value=[
            SearchResult(
                source_type="faq",
                source_id="nocodb_id:1",
                title="Отпуск",
                text="Вопрос: Как оформить отпуск? Ответ: Через отдел кадров.",
                chunk_index=0,
                score=0.9,
            )
        ]
    )

    departments_cache = AsyncMock()
    departments_cache.get_departments = AsyncMock(
        return_value=(["Бухгалтерия"], ["Касса"])
    )

    address_cache = AsyncMock()
    address_cache.ensure_fresh = AsyncMock()
    address_cache.extract_address = MagicMock(return_value=None)

    return AgentLoop(
        llm=llm,
        session_store=session_store,
        pii_parser=pii_parser,
        nocodb_client=nocodb_client,
        qdrant_store=qdrant_store,
        embedder=embedder,
        departments_cache=departments_cache,
        address_cache=address_cache,
    )


# ============================================
# search_internal → Pass 2 (две модели)
# ============================================

async def test_internal_runs_two_passes(agent):
    agent.llm.chat = AsyncMock(
        side_effect=[
            _resp(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="search_internal", args={"query": "отпуск"})],
            ),
            _resp(type="text", content="Отпуск оформляется через отдел кадров.",
                  model="deepseek-v4-flash", tokens=40),
        ]
    )
    result = await process_request_yandex(
        agent=agent, user_id=1, request_text="Как оформить отпуск?", correlation_id="c1",
    )
    assert isinstance(result, TextResponse)
    assert result.answer == "Отпуск оформляется через отдел кадров."
    assert result.tool_used == "search_internal"
    # Pass 1 (lite) + Pass 2 (deepseek).
    assert agent.llm.chat.await_count == 2


async def test_internal_substitutes_hidden_data_for_user(agent):
    """После Pass 2: #ИМЯ → значение в ответе пользователю, плейсхолдер — в Redis."""
    # Чанк с hidden_data: значение скрыто, в тексте плейсхолдер.
    agent.qdrant_store.search = AsyncMock(
        return_value=[
            SearchResult(
                source_type="faq", source_id="nocodb_id:1", title="АХО",
                text="Ответ: пишите на #АХО_КОНТАКТ.",
                chunk_index=0, score=0.95,
                hidden_data="АХО_КОНТАКТ=Bosova.V@mavis.ru",
            )
        ]
    )
    agent.llm.chat = AsyncMock(
        side_effect=[
            _resp(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="search_internal", args={"query": "ахо"})],
            ),
            # deepseek дословно переносит плейсхолдер.
            _resp(type="text", content="Пишите на #АХО_КОНТАКТ.",
                  model="deepseek-v4-flash", tokens=30),
        ]
    )
    result = await process_request_yandex(
        agent=agent, user_id=1, request_text="Контакт АХО?", correlation_id="c2",
    )
    # Пользователь видит реальное значение.
    assert result.answer == "Пишите на Bosova.V@mavis.ru."
    # В Redis уходит МАСКИРОВАННОЕ (с плейсхолдером, без email).
    assistant_call = agent.session_store.append.await_args_list[1]
    assert "#АХО_КОНТАКТ" in assistant_call.kwargs["content"]
    assert "Bosova.V@mavis.ru" not in assistant_call.kwargs["content"]


# ============================================
# bot_command (без Pass 2)
# ============================================

async def test_bot_command_returns_tool_call(agent):
    agent.llm.chat = AsyncMock(
        return_value=_resp(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="search_contacts", args={"query": "Иванов"})],
        )
    )
    result = await process_request_yandex(
        agent=agent, user_id=1, request_text="Найди Иванова", correlation_id="c3",
    )
    assert isinstance(result, ToolCallResponse)
    assert result.tool_calls[0].name == "search_contacts"
    # Только Pass 1, без Pass 2.
    assert agent.llm.chat.await_count == 1


# ============================================
# answer_general → Pass 2 general
# ============================================

async def test_answer_general_runs_pass2(agent):
    agent.llm.chat = AsyncMock(
        side_effect=[
            _resp(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="answer_general", args={})],
            ),
            _resp(type="text", content="Это общий ответ.",
                  model="deepseek-v4-flash", tokens=25),
        ]
    )
    result = await process_request_yandex(
        agent=agent, user_id=1, request_text="Привет, как дела?", correlation_id="c4",
    )
    assert isinstance(result, TextResponse)
    assert result.answer == "Это общий ответ."
    assert result.tool_used == "answer_general"
    assert agent.llm.chat.await_count == 2


# ============================================
# Эскалация: lite вернул текст вместо tool-call
# ============================================

async def test_lite_text_triggers_escalation(agent):
    """Pass 1 (lite) без tool-call → эскалация на сильную модель."""
    agent.llm.chat = AsyncMock(
        side_effect=[
            # lite ошибся — вернул текст.
            _resp(type="text", content="что-то не то", model="yandexgpt-5-lite", tokens=5),
            # эскалация — сильная модель классифицировала.
            _resp(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="search_internal", args={"query": "отпуск"})],
                model="deepseek-v4-flash", tokens=15,
            ),
            # Pass 2.
            _resp(type="text", content="Ответ по контексту.",
                  model="deepseek-v4-flash", tokens=35),
        ]
    )
    result = await process_request_yandex(
        agent=agent, user_id=1, request_text="Как оформить отпуск?", correlation_id="c5",
    )
    assert isinstance(result, TextResponse)
    assert result.answer == "Ответ по контексту."
    # lite + эскалация + Pass 2 = 3 вызова.
    assert agent.llm.chat.await_count == 3


# ============================================
# search_internal пуст → suggest_hr_form
# ============================================

async def test_empty_search_returns_hr_form(agent):
    agent.qdrant_store.search = AsyncMock(return_value=[])
    agent.llm.chat = AsyncMock(
        return_value=_resp(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="search_internal", args={"query": "редкое"})],
        )
    )
    result = await process_request_yandex(
        agent=agent, user_id=1, request_text="Очень редкий вопрос", correlation_id="c6",
    )
    assert isinstance(result, ToolCallResponse)
    assert result.tool_calls[0].name == "suggest_hr_form"
    # Pass 2 не вызывался.
    assert agent.llm.chat.await_count == 1