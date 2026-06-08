"""
Главный мозг агента — двухпроходная схема обработки запроса.

Pass 1: LLM получает запрос + историю + спецификации tools. Решает:
  (a) ответить сама → answer_general
  (b) вызвать agent_internal tool → переход к Pass 2
  (c) вызвать bot_command tool → возврат боту команды

Pass 2: выполняем agent_internal tool, отдаём контекст в LLM, получаем
финальный текстовый ответ.

PII-политика:
  - В Redis всегда сохраняем МАСКИРОВАННЫЕ версии (с [NAME]).
  - Восстановление [NAME] → реальные имена делается только в финальном
    тексте, который уйдёт пользователю.
"""
from dataclasses import dataclass

from app.api.schemas import AskResponse, TextResponse, ToolCall, ToolCallResponse
from app.core.logging import get_logger
from app.llm.base import BaseLLMClient, Message
from app.llm.prompts_gigachat import SYSTEM_PROMPT, make_context_prompt
from app.repositories.nocodb_client import NocoDBClient
from app.services.pii_parser import PiiParser
from app.services.session_store import SessionStore
from app.tools.registry import get_all_tool_specs, is_agent_internal, is_bot_command
from app.tools.tools_internal import execute_internal_tool


logger = get_logger(__name__)


@dataclass
class AgentLoop:
    """
    Композиция зависимостей для двухпроходной обработки запросов.

    Создаётся один раз при старте FastAPI (см. lifespan в app/main.py).
    Все клиенты — долгоживущие.
    """

    llm: BaseLLMClient
    session_store: SessionStore
    pii_parser: PiiParser
    nocodb_client: NocoDBClient


async def process_request(
    agent: AgentLoop,
    user_id: int,
    request_text: str,
    correlation_id: str,
) -> AskResponse:
    """
    Главная функция: входной запрос → ответ (Text или ToolCall).

    Args:
        agent: контейнер с подключёнными клиентами
        user_id: ID пользователя в Telegram
        request_text: оригинальный текст запроса (с реальными именами)
        correlation_id: для логов

    Returns:
        AskResponse — либо TextResponse, либо ToolCallResponse.
    """
    logger.info(
        f"Processing request from user_id={user_id}",
        extra={"correlation_id": correlation_id},
    )

    # 1. Маскируем PII в запросе.
    pii_result = agent.pii_parser.parse(request_text, correlation_id=correlation_id)
    masked_query = pii_result.masked_text
    found_names = pii_result.found_names

    # 2. Достаём историю — она уже хранится в маскированном виде.
    history_dicts = await agent.session_store.get_history(
        user_id=user_id, correlation_id=correlation_id
    )
    history_messages = [
        Message(role=d["role"], content=d["content"]) for d in history_dicts
    ]

    # 3. Pass 1 — собираем messages и шлём в LLM с tools.
    pass1_messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        *history_messages,
        Message(role="user", content=masked_query),
    ]
    llm_response = await agent.llm.chat(
        messages=pass1_messages,
        tools=get_all_tool_specs(),
        correlation_id=correlation_id,
    )

    # 4. LLM ответила текстом — это answer_general.
    if llm_response.type == "text":
        masked_answer = llm_response.content or ""
        final_answer = _restore_pii(masked_answer, found_names)

        await _save_session(
            store=agent.session_store,
            user_id=user_id,
            masked_user_msg=masked_query,
            masked_assistant_msg=masked_answer,
            correlation_id=correlation_id,
        )
        return TextResponse(
            answer=final_answer,
            tool_used="answer_general",
            correlation_id=correlation_id,
        )

    # 5. LLM вызвала tool — берём первый (параллельные tool calls не поддерживаем).
    tool_call = llm_response.tool_calls[0]
    tool_name = tool_call.name
    tool_args = tool_call.args

    logger.info(
        f"LLM called tool: {tool_name}, args={tool_args}",
        extra={"correlation_id": correlation_id},
    )

    # 6. bot_command — возвращаем боту команду, в Redis сохраняем только запрос.
    if is_bot_command(tool_name):
        await agent.session_store.append(
            user_id=user_id,
            role="user",
            content=masked_query,
            correlation_id=correlation_id,
        )
        return ToolCallResponse(
            tool_calls=[ToolCall(name=tool_name, args=tool_args)],
            correlation_id=correlation_id,
        )

    # 7. Неизвестный tool — fallback на answer_general с извинением.
    if not is_agent_internal(tool_name):
        logger.warning(
            f"LLM called unknown tool: {tool_name}",
            extra={"correlation_id": correlation_id},
        )
        fallback = (
            "Не удалось обработать запрос. "
            "Попробуйте переформулировать вопрос."
        )
        await _save_session(
            store=agent.session_store,
            user_id=user_id,
            masked_user_msg=masked_query,
            masked_assistant_msg=fallback,
            correlation_id=correlation_id,
        )
        return TextResponse(
            answer=fallback,
            tool_used="answer_general",
            correlation_id=correlation_id,
        )

    # 8. agent_internal — выполняем tool, получаем контекст.
    context = await execute_internal_tool(
        tool_name=tool_name,
        args=tool_args,
        correlation_id=correlation_id,
    )

    # 9. Pass 2 — LLM формирует финальный ответ из контекста.
    pass2_messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        *history_messages,
        Message(role="user", content=masked_query),
        Message(role="system", content=make_context_prompt(tool_name, context)),
    ]
    pass2_response = await agent.llm.chat(
        messages=pass2_messages,
        tools=None,  # на втором проходе tools не нужны
        correlation_id=correlation_id,
    )

    masked_answer = pass2_response.content or "Не удалось сформировать ответ."
    final_answer = _restore_pii(masked_answer, found_names)

    await _save_session(
        store=agent.session_store,
        user_id=user_id,
        masked_user_msg=masked_query,
        masked_assistant_msg=masked_answer,
        correlation_id=correlation_id,
    )
    return TextResponse(
        answer=final_answer,
        tool_used=tool_name,  # type: ignore[arg-type]
        correlation_id=correlation_id,
    )


def _restore_pii(text: str, original_names: list[str]) -> str:
    """
    Заменить плейсхолдеры [NAME] на оригинальные имена в порядке встречи.

    Если плейсхолдеров больше, чем имён — лишние остаются как [NAME].
    Если имён больше, чем плейсхолдеров — лишние игнорируются (LLM их
    не вернула, это нормально).
    """
    if not original_names:
        return text
    result = text
    for name in original_names:
        if "[NAME]" not in result:
            break
        result = result.replace("[NAME]", name, 1)
    return result


async def _save_session(
    store: SessionStore,
    user_id: int,
    masked_user_msg: str,
    masked_assistant_msg: str,
    correlation_id: str,
) -> None:
    """
    Сохранить пару маскированных сообщений в Redis-сессию.

    В Redis уходят ТОЛЬКО маскированные версии — реальных имён там быть
    не должно. Восстановление PII происходит только в финальном ответе
    пользователю.
    """
    await store.append(
        user_id=user_id,
        role="user",
        content=masked_user_msg,
        correlation_id=correlation_id,
    )
    await store.append(
        user_id=user_id,
        role="assistant",
        content=masked_assistant_msg,
        correlation_id=correlation_id,
    )