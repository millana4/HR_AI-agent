"""
Петля обработки запросов под GigaChat.

Используется как ОСНОВНАЯ, если LLM_PROVIDER=gigachat, и как ЗАПАСНАЯ при
сбое Yandex (fallback). GigaChat сам перебирает модели Max→Pro→Lite внутри
клиента, поэтому параметр model здесь НЕ передаётся.

Схема (двухпроходная, как у Yandex, но проще):
  Pass 1: LLM выбирает действие (tool) или отвечает текстом.
    - answer_general (текстом ИЛИ как tool_call) → общий ответ.
    - search_internal → Qdrant → Pass 2 (ответ по контексту).
    - bot_command (контакты/телефоны/магазины/аптеки/форма) → возврат боту.

PII и хелперы — общие из agent_common.
"""
import asyncio

from app.api.schemas import AskResponse, TextResponse, ToolCall, ToolCallResponse
from app.core.logging import get_logger
from app.llm.base import Message
from app.llm.prompts_gigachat import (
    make_context_prompt,
    make_system_prompt_with_departments,
)
from app.repositories.analytics import save_analytics
from app.services.agent_common import (
    AgentLoop,
    mask_args_for_logs,
    mask_text_for_logs,
    restore_pii,
    restore_pii_in_args,
    save_session,
    strip_service_prefix,
)
from app.services.pii_parser import mask_for_logs
from app.tools.registry import (
    get_all_tool_specs,
    is_agent_general,
    is_agent_internal,
    is_bot_command,
)
from app.tools.tools_internal import NO_CONTEXT, execute_internal_tool


logger = get_logger(__name__)


async def process_request_gigachat(
    agent: AgentLoop,
    user_id: int,
    request_text: str,
    correlation_id: str,
) -> AskResponse:
    """Обработка запроса под GigaChat (основной или fallback-провайдер)."""
    logger.info(
        f"[GC] Processing request from user_id={user_id}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 1. PII-маскирование: реальное имя → NAME_N.
    pii_result = agent.pii_parser.parse(request_text, correlation_id=correlation_id)
    masked_query = pii_result.masked_text
    found_names = pii_result.found_names
    logger.debug(
        f"[GC STEP 1] PII: masked_query={masked_query!r}, "
        f"имён={[mask_for_logs(n) for n in found_names]}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 2. История из Redis (для Pass 2).
    history_dicts = await agent.session_store.get_history(
        user_id=user_id, correlation_id=correlation_id
    )
    history_messages = [
        Message(role=d["role"], content=d["content"]) for d in history_dicts
    ]
    logger.debug(
        f"[GC STEP 2] История: {len(history_messages)} сообщений",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 3. Отделы для системного промпта.
    mavis_deps, votonia_deps = await agent.departments_cache.get_departments(
        agent.nocodb_client, correlation_id=correlation_id
    )
    system_prompt = make_system_prompt_with_departments(mavis_deps, votonia_deps)
    logger.debug(
        f"[GC STEP 3] Отделы: Мавис={len(mavis_deps)}, Вотоня={len(votonia_deps)}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 4. Pass 1 — выбор действия. Историю НЕ подмешиваем.
    #        model НЕ передаём: GigaChat сам перебирает Max→Pro→Lite.
    pass1_messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=masked_query),
    ]
    logger.debug(
        f"[GC STEP 4] Pass 1 → GigaChat. user={masked_query!r}",
        extra={"correlation_id": correlation_id},
    )
    llm_response = await agent.llm.chat(
        messages=pass1_messages,
        tools=get_all_tool_specs(),
        correlation_id=correlation_id,
    )
    logger.debug(
        f"[GC STEP 4] Pass 1 ← GigaChat: type={llm_response.type}, "
        f"content={llm_response.content!r}, "
        f"tool_calls={[(tc.name, tc.args) for tc in llm_response.tool_calls]}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 5. answer_general как ТЕКСТ: LLM ответила текстом без tool.
    if llm_response.type == "text":
        return await _general_answer(
            agent, user_id, masked_query, found_names,
            strip_service_prefix(llm_response.content or ""),
            correlation_id,
        )

    # ШАГ 6. LLM вызвала tool. Берём первый.
    tool_call = llm_response.tool_calls[0]
    tool_name = tool_call.name
    tool_args = tool_call.args
    logger.info(
        f"[GC STEP 6] Выбран tool: {tool_name}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 7. bot_command (контакты/телефоны/магазины/аптеки/форма).
    if is_bot_command(tool_name):
        # Для магазинов/аптек извлекаем топоним из словаря адресов.
        if tool_name in ("search_shop", "search_drugstore"):
            await agent.address_cache.ensure_fresh(
                agent.nocodb_client, correlation_id=correlation_id
            )
            topo = agent.address_cache.extract_address(masked_query)
            if topo:
                tool_args = {**tool_args, "query": topo}
                logger.debug(
                    f"[GC STEP 7] Адрес из словаря: {topo!r}",
                    extra={"correlation_id": correlation_id},
                )

        restored_args = restore_pii_in_args(tool_args, found_names)
        logger.debug(
            f"[GC STEP 7] bot_command {tool_name}: "
            f"args (маскировано)={mask_args_for_logs(restored_args, found_names)}",
            extra={"correlation_id": correlation_id},
        )
        await asyncio.create_task(save_analytics(
            nocodb_client=agent.nocodb_client,
            user_id=user_id,
            masked_question=masked_query,
            masked_answer="",
            tool_used=tool_name,
            correlation_id=correlation_id,
        ))
        return ToolCallResponse(
            tool_calls=[ToolCall(name=tool_name, args=restored_args)],
            correlation_id=correlation_id,
        )

    # ШАГ 8. answer_general как TOOL_CALL: общий ответ.
    if is_agent_general(tool_name):
        logger.info(
            "[GC STEP 8] answer_general (tool_call) → общий ответ",
            extra={"correlation_id": correlation_id},
        )
        # Повторный вызов без tools — получаем текст из общих знаний.
        general_messages = [
            Message(role="system", content=system_prompt),
            *history_messages,
            Message(role="user", content=masked_query),
        ]
        general_response = await agent.llm.chat(
            messages=general_messages,
            tools=None,
            correlation_id=correlation_id,
        )
        return await _general_answer(
            agent, user_id, masked_query, found_names,
            strip_service_prefix(general_response.content or ""),
            correlation_id,
        )

    # ШАГ 9. search_internal — поиск в Qdrant, затем Pass 2.
    if is_agent_internal(tool_name):
        context = await execute_internal_tool(
            tool_name=tool_name,
            args=tool_args,
            qdrant_store=agent.qdrant_store,
            embedder=agent.embedder,
            correlation_id=correlation_id,
        )
        logger.debug(
            f"[GC STEP 9] search_internal: контекст {len(context)} симв.",
            extra={"correlation_id": correlation_id},
        )
        if context == NO_CONTEXT:
            logger.info(
                "[GC STEP 9] Поиск пуст — предлагаем форму HR",
                extra={"correlation_id": correlation_id},
            )
            await asyncio.create_task(save_analytics(
                nocodb_client=agent.nocodb_client,
                user_id=user_id,
                masked_question=masked_query,
                masked_answer="",
                tool_used="suggest_hr_form",
                correlation_id=correlation_id,
            ))
            return ToolCallResponse(
                tool_calls=[ToolCall(name="suggest_hr_form", args={})],
                correlation_id=correlation_id,
            )

        # Pass 2 — ответ по контексту. model НЕ передаём (стратегия GigaChat).
        pass2_system = make_context_prompt(tool_name, context)
        pass2_messages = [
            Message(role="system", content=pass2_system),
            *history_messages,
            Message(role="user", content=masked_query),
        ]
        logger.debug(
            f"[GC STEP 11] Pass 2 → GigaChat. user={masked_query!r}",
            extra={"correlation_id": correlation_id},
        )
        pass2_response = await agent.llm.chat(
            messages=pass2_messages,
            tools=None,
            correlation_id=correlation_id,
        )
        masked_answer = strip_service_prefix(
            pass2_response.content or ""
        ) or "Не удалось сформировать ответ."
        final_answer = restore_pii(masked_answer, found_names)
        logger.debug(
            f"[GC STEP 12] Pass 2 ← GigaChat. "
            f"answer={mask_text_for_logs(final_answer, found_names)!r}",
            extra={"correlation_id": correlation_id},
        )
        await save_session(
            store=agent.session_store,
            user_id=user_id,
            masked_user_msg=masked_query,
            masked_assistant_msg=masked_answer,
            correlation_id=correlation_id,
        )
        await asyncio.create_task(save_analytics(
            nocodb_client=agent.nocodb_client,
            user_id=user_id,
            masked_question=masked_query,
            masked_answer=masked_answer,
            tool_used=tool_name,
            correlation_id=correlation_id,
        ))
        return TextResponse(
            answer=final_answer,
            tool_used=tool_name,  # type: ignore[arg-type]
            correlation_id=correlation_id,
        )

    # ШАГ 10. Неизвестный tool — защита.
    logger.warning(
        f"[GC STEP 10] Неизвестный tool: {tool_name} — fallback-заглушка",
        extra={"correlation_id": correlation_id},
    )
    fallback = "Не удалось обработать запрос. Попробуйте переформулировать вопрос."
    return TextResponse(
        answer=fallback,
        tool_used="answer_general",
        correlation_id=correlation_id,
    )


async def _general_answer(
    agent: AgentLoop,
    user_id: int,
    masked_query: str,
    found_names: list[str],
    masked_answer: str,
    correlation_id: str,
) -> AskResponse:
    """Финализировать общий ответ (answer_general): PII, сессия, аналитика."""
    masked_answer = masked_answer or "Не удалось сформировать ответ."
    final_answer = restore_pii(masked_answer, found_names)
    logger.debug(
        f"[GC] answer_general → {mask_text_for_logs(final_answer, found_names)!r}",
        extra={"correlation_id": correlation_id},
    )
    await save_session(
        store=agent.session_store,
        user_id=user_id,
        masked_user_msg=masked_query,
        masked_assistant_msg=masked_answer,
        correlation_id=correlation_id,
    )
    await asyncio.create_task(save_analytics(
        nocodb_client=agent.nocodb_client,
        user_id=user_id,
        masked_question=masked_query,
        masked_answer=masked_answer,
        tool_used="answer_general",
        correlation_id=correlation_id,
    ))
    return TextResponse(
        answer=final_answer,
        tool_used="answer_general",
        correlation_id=correlation_id,
    )