"""
Интеграционные тесты agent_loop с моками всех внешних зависимостей.

Проверяем главную логику двухпроходной схемы:
  - Pass 1 → text (answer_general)
  - Pass 1 → bot_command
  - Pass 1 → search_internal → Pass 2
  - search_internal пуст → suggest_hr_form
  - PII-маскирование и восстановление
  - Подмешивание истории
  - Fallback на неизвестный tool

Не дёргаем FastAPI и HTTP — тестируем чистую функцию process_request.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.schemas import TextResponse, ToolCallResponse
from app.llm.base import LLMResponse, ToolCall as LLMToolCall
from app.rag.qdrant_store import SearchResult
from app.services.agent_loop import AgentLoop, process_request
from app.services.pii_parser import PiiParseResult


# ============================================
# Фикстуры
# ============================================

def _make_pii_result(masked_text: str, found_names: list[str]) -> PiiParseResult:
    return PiiParseResult(masked_text=masked_text, found_names=found_names)


@pytest.fixture
def agent():
    """AgentLoop со всеми моками."""
    llm = AsyncMock()
    session_store = AsyncMock()
    session_store.get_history = AsyncMock(return_value=[])
    session_store.append = AsyncMock()

    pii_parser = MagicMock()
    pii_parser.parse = MagicMock(
        side_effect=lambda text, correlation_id="-": _make_pii_result(text, [])
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
                title="Вопрос про отпуск",
                text="Вопрос: Как оформить отпуск? Ответ: Через отдел кадров.",
                chunk_index=0,
                score=0.9,
            )
        ]
    )

    return AgentLoop(
        llm=llm,
        session_store=session_store,
        pii_parser=pii_parser,
        nocodb_client=nocodb_client,
        qdrant_store=qdrant_store,
        embedder=embedder,
    )


# ============================================
# Сценарий 1: Pass 1 → text (answer_general)
# ============================================

async def test_pass1_text_returns_answer_general(agent):
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="Привет! Чем помочь?")
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Привет", correlation_id="cid-1",
    )
    assert isinstance(result, TextResponse)
    assert result.answer == "Привет! Чем помочь?"
    assert result.tool_used == "answer_general"
    assert result.correlation_id == "cid-1"


async def test_pass1_text_saves_masked_pair_to_redis(agent):
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="Ответ.")
    )
    await process_request(
        agent=agent, user_id=123, request_text="Запрос", correlation_id="cid-1",
    )
    assert agent.session_store.append.await_count == 2
    user_call = agent.session_store.append.await_args_list[0]
    assistant_call = agent.session_store.append.await_args_list[1]
    assert user_call.kwargs["role"] == "user"
    assert user_call.kwargs["content"] == "Запрос"
    assert assistant_call.kwargs["role"] == "assistant"
    assert assistant_call.kwargs["content"] == "Ответ."


# ============================================
# Сценарий 2: Pass 1 → bot_command
# ============================================

async def test_pass1_bot_command_returns_tool_call_response(agent):
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="search_contacts", args={"query": "Иванов"})],
        )
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Найди Иванова", correlation_id="cid-2",
    )
    assert isinstance(result, ToolCallResponse)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_contacts"
    assert result.tool_calls[0].args == {"query": "Иванов"}
    assert result.correlation_id == "cid-2"


async def test_pass1_bot_command_saves_only_user_message(agent):
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="search_contacts", args={"query": "Иванов"})],
        )
    )
    await process_request(
        agent=agent, user_id=123, request_text="Найди Иванова", correlation_id="cid-2",
    )
    assert agent.session_store.append.await_count == 1
    call = agent.session_store.append.await_args_list[0]
    assert call.kwargs["role"] == "user"
    assert call.kwargs["content"] == "Найди Иванова"


async def test_pass1_bot_command_does_not_call_pass2(agent):
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="search_shop", args={"query": "СПб Невский"})],
        )
    )
    await process_request(
        agent=agent, user_id=123, request_text="Найди магазин", correlation_id="cid-2",
    )
    assert agent.llm.chat.await_count == 1


async def test_bot_command_restores_real_name_in_args(agent):
    """В args bot_command [NAME] заменяется на реальную фамилию для бота."""
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result("Найди телефон [NAME]", ["Иванов"])
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="search_contacts", args={"query": "[NAME]"})],
        )
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Найди телефон Иванова", correlation_id="cid-bc",
    )
    assert isinstance(result, ToolCallResponse)
    assert result.tool_calls[0].args == {"query": "Иванов"}


# ============================================
# Сценарий 3: Pass 1 → search_internal → Pass 2
# ============================================

async def test_agent_internal_runs_two_passes(agent):
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="search_internal", args={"query": "отпуск"})],
            ),
            LLMResponse(type="text", content="Отпуск оформляется так-то."),
        ]
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Как оформить отпуск?", correlation_id="cid-3",
    )
    assert isinstance(result, TextResponse)
    assert result.answer == "Отпуск оформляется так-то."
    assert result.tool_used == "search_internal"
    assert agent.llm.chat.await_count == 2


async def test_agent_internal_pass2_receives_tool_context(agent):
    """Pass 2 получает контекст в system-сообщении (склеен с SYSTEM_PROMPT)."""
    agent.qdrant_store.search = AsyncMock(
        return_value=[
            SearchResult(
                source_type="faq",
                source_id="nocodb_id:7",
                title="График работы",
                text="Вопрос: Какой график? Ответ: с 9 до 18.",
                chunk_index=0,
                score=0.95,
            )
        ]
    )
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="search_internal", args={"query": "график"})],
            ),
            LLMResponse(type="text", content="Финальный ответ."),
        ]
    )
    await process_request(
        agent=agent, user_id=123, request_text="Какой график?", correlation_id="cid-3",
    )
    pass2_call = agent.llm.chat.await_args_list[1]
    pass2_messages = pass2_call.kwargs["messages"]
    # Контекст теперь в system-сообщении (первое), а не в последнем.
    system_content = pass2_messages[0].content
    assert pass2_messages[0].role == "system"
    assert "search_internal" in system_content
    assert "с 9 до 18" in system_content
    assert pass2_call.kwargs["tools"] is None
    # Запрос ушёл в эмбеддер.
    agent.embedder.embed_query.assert_awaited_once()
    assert agent.embedder.embed_query.await_args.args[0] == "график"


async def test_agent_internal_saves_masked_assistant_to_redis(agent):
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result("Запрос про [NAME]", ["Иванов"])
    )
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="search_internal", args={"query": "[NAME]"})],
            ),
            LLMResponse(type="text", content="Обратитесь к [NAME] из отдела."),
        ]
    )
    await process_request(
        agent=agent, user_id=123, request_text="Где работает Иванов?", correlation_id="cid-3",
    )
    assistant_call = agent.session_store.append.await_args_list[1]
    assert assistant_call.kwargs["role"] == "assistant"
    assert "[NAME]" in assistant_call.kwargs["content"]
    assert "Иванов" not in assistant_call.kwargs["content"]


async def test_empty_search_returns_hr_form(agent):
    """Если search_internal ничего не нашёл → bot_command suggest_hr_form."""
    agent.qdrant_store.search = AsyncMock(return_value=[])
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="search_internal", args={"query": "что-то редкое"})],
        )
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Очень редкий вопрос", correlation_id="cid-hr",
    )
    assert isinstance(result, ToolCallResponse)
    assert result.tool_calls[0].name == "suggest_hr_form"
    assert result.tool_calls[0].args == {}
    # Pass 2 не вызывался — только Pass 1.
    assert agent.llm.chat.await_count == 1


# ============================================
# Сценарий 4: PII-восстановление
# ============================================

async def test_pii_restored_in_final_text(agent):
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result("Где работает [NAME]?", ["Иванов"])
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="[NAME] работает в отделе закупок.")
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Где работает Иванов?", correlation_id="cid-4",
    )
    assert "Иванов работает в отделе закупок." == result.answer
    assert "[NAME]" not in result.answer


async def test_pii_multiple_names_restored_in_order(agent):
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result(
            "[NAME] и [NAME] работают вместе", ["Иванов", "Петров"],
        )
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="text", content="[NAME] и [NAME] работают в одном отделе.",
        )
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Иванов и Петров вместе работают?", correlation_id="cid-4",
    )
    assert result.answer == "Иванов и Петров работают в одном отделе."


async def test_pii_redis_stores_masked_user_message(agent):
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result("Найди [NAME]", ["Иванов"])
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="Не нашёл.")
    )
    await process_request(
        agent=agent, user_id=123, request_text="Найди Иванова", correlation_id="cid-4",
    )
    user_call = agent.session_store.append.await_args_list[0]
    assert user_call.kwargs["content"] == "Найди [NAME]"
    assert "Иванов" not in user_call.kwargs["content"]


# ============================================
# История подмешивается в messages
# ============================================

async def test_history_is_included_in_pass1_messages(agent):
    agent.session_store.get_history = AsyncMock(
        return_value=[
            {"role": "user", "content": "Предыдущий вопрос"},
            {"role": "assistant", "content": "Предыдущий ответ"},
        ]
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="Новый ответ.")
    )
    await process_request(
        agent=agent, user_id=123, request_text="Новый вопрос", correlation_id="cid-h",
    )
    pass1_call = agent.llm.chat.await_args_list[0]
    messages = pass1_call.kwargs["messages"]
    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[1].content == "Предыдущий вопрос"
    assert messages[2].content == "Предыдущий ответ"
    assert messages[3].content == "Новый вопрос"


# ============================================
# Fallback на неизвестный tool
# ============================================

async def test_unknown_tool_falls_back_to_answer_general(agent):
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls", content="",
            tool_calls=[LLMToolCall(name="unknown_tool", args={})],
        )
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Что-то странное", correlation_id="cid-fb",
    )
    assert isinstance(result, TextResponse)
    assert result.tool_used == "answer_general"
    assert "переформулировать" in result.answer.lower() or "не удалось" in result.answer.lower()


# ============================================
# Пустой ответ Pass 2
# ============================================

async def test_empty_pass2_response_has_fallback_text(agent):
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls", content="",
                tool_calls=[LLMToolCall(name="search_internal", args={"query": "x"})],
            ),
            LLMResponse(type="text", content=""),
        ]
    )
    result = await process_request(
        agent=agent, user_id=123, request_text="Что угодно", correlation_id="cid-e",
    )
    assert isinstance(result, TextResponse)
    assert result.answer