"""
Интеграционные тесты agent_loop с моками всех внешних зависимостей.

Проверяем главную логику двухпроходной схемы:
  - Pass 1 → text (answer_general)
  - Pass 1 → bot_command
  - Pass 1 → agent_internal → Pass 2
  - PII-маскирование и восстановление
  - Подмешивание истории
  - Fallback на неизвестный tool

Не дёргаем FastAPI и HTTP — тестируем чистую функцию process_request.
Реальные тесты HTTP-роута будут в шаге 17 (E2E).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.schemas import TextResponse, ToolCallResponse
from app.llm.base import LLMResponse, ToolCall as LLMToolCall
from app.services.agent_loop import AgentLoop, process_request
from app.services.pii_parser import PiiParseResult


# ============================================
# Фикстуры
# ============================================

def _make_pii_result(masked_text: str, found_names: list[str]) -> PiiParseResult:
    """Создать PiiParseResult — совместимый с реальной реализацией."""
    return PiiParseResult(masked_text=masked_text, found_names=found_names)


@pytest.fixture
def agent():
    """
    AgentLoop со всеми моками. Конкретное поведение настраивается
    в каждом тесте через side_effect / return_value.
    """
    llm = AsyncMock()
    session_store = AsyncMock()
    session_store.get_history = AsyncMock(return_value=[])
    session_store.append = AsyncMock()

    pii_parser = MagicMock()
    # По умолчанию ничего не маскируем — тесты, которым нужен PII,
    # переопределяют это сами.
    pii_parser.parse = MagicMock(
        side_effect=lambda text, correlation_id="-": _make_pii_result(text, [])
    )

    nocodb_client = MagicMock()

    return AgentLoop(
        llm=llm,
        session_store=session_store,
        pii_parser=pii_parser,
        nocodb_client=nocodb_client,
    )


# ============================================
# Сценарий 1: Pass 1 → text (answer_general)
# ============================================

async def test_pass1_text_returns_answer_general(agent):
    """LLM вернула текст без tool_call → answer_general."""
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="Привет! Чем помочь?")
    )

    result = await process_request(
        agent=agent,
        user_id=123,
        request_text="Привет",
        correlation_id="cid-1",
    )

    assert isinstance(result, TextResponse)
    assert result.answer == "Привет! Чем помочь?"
    assert result.tool_used == "answer_general"
    assert result.correlation_id == "cid-1"


async def test_pass1_text_saves_masked_pair_to_redis(agent):
    """Pass 1 text → в Redis ушли user + assistant сообщения."""
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="Ответ.")
    )

    await process_request(
        agent=agent,
        user_id=123,
        request_text="Запрос",
        correlation_id="cid-1",
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
    """LLM вызвала search_contacts → ToolCallResponse боту."""
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls",
            content="",
            tool_calls=[
                LLMToolCall(name="search_contacts", args={"query": "Иванов"})
            ],
        )
    )

    result = await process_request(
        agent=agent,
        user_id=123,
        request_text="Найди Иванова",
        correlation_id="cid-2",
    )

    assert isinstance(result, ToolCallResponse)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_contacts"
    assert result.tool_calls[0].args == {"query": "Иванов"}
    assert result.correlation_id == "cid-2"


async def test_pass1_bot_command_saves_only_user_message(agent):
    """Для bot_command в Redis сохраняем только запрос пользователя."""
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls",
            content="",
            tool_calls=[
                LLMToolCall(name="search_contacts", args={"query": "Иванов"})
            ],
        )
    )

    await process_request(
        agent=agent,
        user_id=123,
        request_text="Найди Иванова",
        correlation_id="cid-2",
    )

    assert agent.session_store.append.await_count == 1
    call = agent.session_store.append.await_args_list[0]
    assert call.kwargs["role"] == "user"
    assert call.kwargs["content"] == "Найди Иванова"


async def test_pass1_bot_command_does_not_call_pass2(agent):
    """bot_command завершает работу — Pass 2 не выполняется."""
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls",
            content="",
            tool_calls=[
                LLMToolCall(name="search_shop", args={"query": "СПб Невский"})
            ],
        )
    )

    await process_request(
        agent=agent,
        user_id=123,
        request_text="Найди магазин",
        correlation_id="cid-2",
    )

    # LLM вызвалась один раз (только Pass 1)
    assert agent.llm.chat.await_count == 1


# ============================================
# Сценарий 3: Pass 1 → agent_internal → Pass 2
# ============================================

async def test_agent_internal_runs_two_passes(agent):
    """search_faq → tool выполняется → Pass 2 формирует финальный ответ."""
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls",
                content="",
                tool_calls=[
                    LLMToolCall(name="search_faq", args={"query": "отпуск"})
                ],
            ),
            LLMResponse(type="text", content="Отпуск оформляется так-то."),
        ]
    )

    result = await process_request(
        agent=agent,
        user_id=123,
        request_text="Как оформить отпуск?",
        correlation_id="cid-3",
    )

    assert isinstance(result, TextResponse)
    assert result.answer == "Отпуск оформляется так-то."
    assert result.tool_used == "search_faq"
    assert agent.llm.chat.await_count == 2


async def test_agent_internal_pass2_receives_tool_context(agent):
    """Pass 2 получает messages с make_context_prompt от tool."""
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls",
                content="",
                tool_calls=[
                    LLMToolCall(name="search_faq", args={"query": "график"})
                ],
            ),
            LLMResponse(type="text", content="Финальный ответ."),
        ]
    )

    await process_request(
        agent=agent,
        user_id=123,
        request_text="Какой график?",
        correlation_id="cid-3",
    )

    # Второй вызов LLM — Pass 2 с контекстом от tool
    pass2_call = agent.llm.chat.await_args_list[1]
    pass2_messages = pass2_call.kwargs["messages"]
    # Последнее сообщение должно содержать упоминание tool и заглушку
    last_msg_content = pass2_messages[-1].content
    assert "search_faq" in last_msg_content
    assert "график" in last_msg_content
    # На втором проходе tools=None
    assert pass2_call.kwargs["tools"] is None


async def test_agent_internal_saves_masked_assistant_to_redis(agent):
    """В Redis уходит маскированная версия ответа Pass 2, не финальная."""
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result("Запрос про [NAME]", ["Иванов"])
    )
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls",
                content="",
                tool_calls=[
                    LLMToolCall(name="search_faq", args={"query": "[NAME]"})
                ],
            ),
            LLMResponse(type="text", content="Обратитесь к [NAME] из отдела."),
        ]
    )

    await process_request(
        agent=agent,
        user_id=123,
        request_text="Где работает Иванов?",
        correlation_id="cid-3",
    )

    # Assistant в Redis — с [NAME], а не с восстановленным "Иванов"
    assistant_call = agent.session_store.append.await_args_list[1]
    assert assistant_call.kwargs["role"] == "assistant"
    assert "[NAME]" in assistant_call.kwargs["content"]
    assert "Иванов" not in assistant_call.kwargs["content"]


# ============================================
# Сценарий 4: PII-восстановление
# ============================================

async def test_pii_restored_in_final_text(agent):
    """[NAME] в ответе LLM заменяется на оригинальное имя."""
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result("Где работает [NAME]?", ["Иванов"])
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="text",
            content="[NAME] работает в отделе закупок.",
        )
    )

    result = await process_request(
        agent=agent,
        user_id=123,
        request_text="Где работает Иванов?",
        correlation_id="cid-4",
    )

    assert "Иванов работает в отделе закупок." == result.answer
    assert "[NAME]" not in result.answer


async def test_pii_multiple_names_restored_in_order(agent):
    """Несколько [NAME] заменяются в порядке встречи found_names."""
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result(
            "[NAME] и [NAME] работают вместе",
            ["Иванов", "Петров"],
        )
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="text",
            content="[NAME] и [NAME] работают в одном отделе.",
        )
    )

    result = await process_request(
        agent=agent,
        user_id=123,
        request_text="Иванов и Петров вместе работают?",
        correlation_id="cid-4",
    )

    assert result.answer == "Иванов и Петров работают в одном отделе."


async def test_pii_redis_stores_masked_user_message(agent):
    """В Redis user-сообщение всегда маскированное, никаких реальных имён."""
    agent.pii_parser.parse = MagicMock(
        return_value=_make_pii_result("Найди [NAME]", ["Иванов"])
    )
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(type="text", content="Не нашёл.")
    )

    await process_request(
        agent=agent,
        user_id=123,
        request_text="Найди Иванова",
        correlation_id="cid-4",
    )

    user_call = agent.session_store.append.await_args_list[0]
    assert user_call.kwargs["content"] == "Найди [NAME]"
    assert "Иванов" not in user_call.kwargs["content"]


# ============================================
# История подмешивается в messages
# ============================================

async def test_history_is_included_in_pass1_messages(agent):
    """История из Redis превращается в Message-объекты и подмешивается."""
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
        agent=agent,
        user_id=123,
        request_text="Новый вопрос",
        correlation_id="cid-h",
    )

    pass1_call = agent.llm.chat.await_args_list[0]
    messages = pass1_call.kwargs["messages"]
    # system + 2 истории + 1 новый user = 4
    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[1].content == "Предыдущий вопрос"
    assert messages[2].content == "Предыдущий ответ"
    assert messages[3].content == "Новый вопрос"


# ============================================
# Fallback на неизвестный tool
# ============================================

async def test_unknown_tool_falls_back_to_answer_general(agent):
    """Если LLM вернула неизвестный tool — отвечаем извинением."""
    agent.llm.chat = AsyncMock(
        return_value=LLMResponse(
            type="tool_calls",
            content="",
            tool_calls=[LLMToolCall(name="unknown_tool", args={})],
        )
    )

    result = await process_request(
        agent=agent,
        user_id=123,
        request_text="Что-то странное",
        correlation_id="cid-fb",
    )

    assert isinstance(result, TextResponse)
    assert result.tool_used == "answer_general"
    assert "переформулировать" in result.answer.lower() or "не удалось" in result.answer.lower()


# ============================================
# Пустой ответ Pass 2
# ============================================

async def test_empty_pass2_response_has_fallback_text(agent):
    """Если Pass 2 вернул пустой content — отдаём заглушку."""
    agent.llm.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                type="tool_calls",
                content="",
                tool_calls=[
                    LLMToolCall(name="search_faq", args={"query": "x"})
                ],
            ),
            LLMResponse(type="text", content=""),
        ]
    )

    result = await process_request(
        agent=agent,
        user_id=123,
        request_text="Что угодно",
        correlation_id="cid-e",
    )

    assert isinstance(result, TextResponse)
    assert result.answer  # непустая строка