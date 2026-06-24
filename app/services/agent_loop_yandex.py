"""
Петля обработки запросов под Yandex AI Studio.

Отличие от GigaChat-петли (agent_loop.py): здесь Pass 1 (lite) — ЧИСТЫЙ
КЛАССИФИКАТОР. Он ВСЕГДА возвращает tool-call (включая answer_general),
сам текст ответа не пишет. Ответ формирует Pass 2 (deepseek).

Схема:
  Pass 1 (yandexgpt-5-lite): выбор tool из 8 вариантов.
    - search_contacts / search_ats_* / search_shop / search_drugstore /
      suggest_hr_form  → bot_command: вернуть боту, обработка завершена.
    - search_internal  → векторный поиск в Qdrant → Pass 2 (deepseek с чанками).
    - answer_general   → Pass 2 (deepseek БЕЗ Qdrant, ответ из общих знаний).

  Если Pass 1 вернул текст вместо tool-call — это сбой классификации lite
  (сигнал, что пора поднять модель Pass 1). Отдаём вежливую заглушку.

PII и аналитика — как в GigaChat-петле (общие хелперы в agent_common).
Нумерация [STEP N] своя, с пометкой YA.
"""
import asyncio

from app.api.schemas import AskResponse, TextResponse, ToolCall, ToolCallResponse
from app.core.config import Config
from app.core.logging import get_logger
from app.llm.base import Message
from app.llm.prompts_yandex import (
    make_system_prompt_with_departments,
    make_context_prompt,
    make_general_prompt,
)
from app.repositories.analytics import save_analytics
from app.services.agent_common import (
    mask_args_for_logs,
    mask_text_for_logs,
    restore_pii,
    restore_pii_in_args,
    save_session,
    strip_service_prefix,
)
from app.services.agent_loop import AgentLoop  # переиспользуем тот же контейнер
from app.services.pii_parser import mask_for_logs
from app.tools.registry import (
    get_all_tool_specs,
    is_agent_general,
    is_agent_internal,
    is_bot_command,
)
from app.tools.tools_internal import NO_CONTEXT, execute_internal_tool


logger = get_logger(__name__)


async def process_request_yandex(
    agent: AgentLoop,
    user_id: int,
    request_text: str,
    correlation_id: str,
) -> AskResponse:
    """Обработка запроса под Yandex: lite-классификатор + deepseek-ответчик."""
    logger.info(
        f"[YA] Processing request from user_id={user_id}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 1. Маскируем ПД: реальное имя → NAME_N.
    pii_result = agent.pii_parser.parse(request_text, correlation_id=correlation_id)
    masked_query = pii_result.masked_text
    found_names = pii_result.found_names
    logger.debug(
        f"[YA STEP 1] PII: masked_query={masked_query!r}, "
        f"имён={[mask_for_logs(n) for n in found_names]}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 2. История из Redis (понадобится на Pass 2).
    history_dicts = await agent.session_store.get_history(
        user_id=user_id, correlation_id=correlation_id
    )
    history_messages = [
        Message(role=d["role"], content=d["content"]) for d in history_dicts
    ]
    logger.debug(
        f"[YA STEP 2] История: {len(history_messages)} сообщений",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 3. Списки отделов для промпта Pass 1.
    mavis_deps, votonia_deps = await agent.departments_cache.get_departments(
        agent.nocodb_client, correlation_id=correlation_id
    )
    system_prompt = make_system_prompt_with_departments(mavis_deps, votonia_deps)
    logger.debug(
        f"[YA STEP 3] Отделы: Мавис={len(mavis_deps)}, Вотоня={len(votonia_deps)}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 4. Pass 1 — классификация (lite). Историю НЕ подмешиваем.
    pass1_messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=masked_query),
    ]
    logger.debug(
        f"[YA STEP 4] Pass 1 → lite. user={masked_query!r}",
        extra={"correlation_id": correlation_id},
    )
    llm_response = await agent.llm.chat(
        messages=pass1_messages,
        tools=get_all_tool_specs(),
        correlation_id=correlation_id,
        model=Config.YANDEX_MODEL_PASS1,
    )
    logger.debug(
        f"[YA STEP 4] Pass 1 ← lite: type={llm_response.type}, "
        f"tool_calls={[(tc.name, tc.args) for tc in llm_response.tool_calls]}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 5. Lite вернула текст вместо tool-call — сбой классификации.
    #        Эскалируем: повторяем Pass 1 на более сильной модели (PASS2 = deepseek).
    if llm_response.type == "text" or not llm_response.tool_calls:
        logger.warning(
            f"[YA STEP 5] Pass 1 (lite) не вернул tool-call — эскалация на "
            f"{Config.YANDEX_MODEL_PASS2}",
            extra={"correlation_id": correlation_id},
        )
        llm_response = await agent.llm.chat(
            messages=pass1_messages,
            tools=get_all_tool_specs(),
            correlation_id=correlation_id,
            model=Config.YANDEX_MODEL_PASS2,
        )
        logger.debug(
            f"[YA STEP 5] Эскалация ← {Config.YANDEX_MODEL_PASS2}: "
            f"type={llm_response.type}, "
            f"tool_calls={[(tc.name, tc.args) for tc in llm_response.tool_calls]}",
            extra={"correlation_id": correlation_id},
        )
        # Если и сильная модель не классифицировала — трактуем как общий вопрос.
        if llm_response.type == "text" or not llm_response.tool_calls:
            logger.warning(
                "[YA STEP 5] Эскалация тоже без tool-call — трактуем как answer_general",
                extra={"correlation_id": correlation_id},
            )
            return await _pass2_general(
                agent, user_id, masked_query, found_names,
                history_messages, correlation_id,
            )

    # ШАГ 6. Берём первый tool-call (множественные — отдельный шаг плана).
    tool_call = llm_response.tool_calls[0]
    tool_name = tool_call.name
    tool_args = tool_call.args
    logger.info(
        f"[YA STEP 6] Pass 1 выбрал tool: {tool_name}",
        extra={"correlation_id": correlation_id},
    )

    # ШАГ 7. bot_command (контакты, телефоны, магазины, аптеки, форма HR).
    #        Возвращаем боту команду. ПД восстанавливаем в args.
    if is_bot_command(tool_name):
        restored_args = restore_pii_in_args(tool_args, found_names)
        logger.debug(
            f"[YA STEP 7] bot_command {tool_name}: "
            f"args (маскировано)={mask_args_for_logs(restored_args, found_names)}",
            extra={"correlation_id": correlation_id},
        )
        asyncio.create_task(save_analytics(
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

    # ШАГ 8. answer_general — общий вопрос. Pass 2 на deepseek БЕЗ Qdrant.
    if is_agent_general(tool_name):
        logger.info(
            "[YA STEP 8] answer_general → Pass 2 deepseek (без векторного поиска)",
            extra={"correlation_id": correlation_id},
        )
        return await _pass2_general(
            agent, user_id, masked_query, found_names,
            history_messages, correlation_id,
        )

    # ШАГ 9. search_internal — векторный поиск, затем Pass 2 с контекстом.
    if is_agent_internal(tool_name):
        context = await execute_internal_tool(
            tool_name=tool_name,
            args=tool_args,
            qdrant_store=agent.qdrant_store,
            embedder=agent.embedder,
            correlation_id=correlation_id,
        )
        logger.debug(
            f"[YA STEP 9] search_internal вернул контекст ({len(context)} симв.)",
            extra={"correlation_id": correlation_id},
        )

        # Поиск пуст → предлагаем форму HR.
        if context == NO_CONTEXT:
            logger.info(
                "[YA STEP 9] search_internal пусто — предлагаем форму HR",
                extra={"correlation_id": correlation_id},
            )
            asyncio.create_task(save_analytics(
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

        return await _pass2_internal(
            agent, user_id, masked_query, found_names,
            history_messages, tool_name, context, correlation_id,
        )

    # ШАГ 10. Неизвестный tool — защита.
    logger.warning(
        f"[YA STEP 10] Неизвестный tool: {tool_name} — fallback",
        extra={"correlation_id": correlation_id},
    )
    fallback = "Не удалось обработать запрос. Попробуйте переформулировать вопрос."
    return TextResponse(
        answer=fallback,
        tool_used="answer_general",
        correlation_id=correlation_id,
    )


async def _pass2_general(
    agent: AgentLoop,
    user_id: int,
    masked_query: str,
    found_names: list[str],
    history_messages: list[Message],
    correlation_id: str,
) -> AskResponse:
    """Pass 2 для answer_general: deepseek отвечает из общих знаний, без Qdrant."""
    pass2_messages = [
        Message(role="system", content=make_general_prompt()),
        *history_messages,
        Message(role="user", content=masked_query),
    ]
    logger.debug(
        f"[YA STEP 11] Pass 2 (general) → deepseek. user={masked_query!r}",
        extra={"correlation_id": correlation_id},
    )
    pass2_response = await agent.llm.chat(
        messages=pass2_messages,
        tools=None,
        correlation_id=correlation_id,
        model=Config.YANDEX_MODEL_PASS2,
    )
    masked_answer = strip_service_prefix(
        pass2_response.content or ""
    ) or "Не удалось сформировать ответ."
    final_answer = restore_pii(masked_answer, found_names)
    logger.debug(
        f"[YA STEP 12] Pass 2 (general) ← deepseek. "
        f"answer (маскировано)={mask_text_for_logs(final_answer, found_names)!r}",
        extra={"correlation_id": correlation_id},
    )
    await save_session(
        store=agent.session_store,
        user_id=user_id,
        masked_user_msg=masked_query,
        masked_assistant_msg=masked_answer,
        correlation_id=correlation_id,
    )
    asyncio.create_task(save_analytics(
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


async def _pass2_internal(
    agent: AgentLoop,
    user_id: int,
    masked_query: str,
    found_names: list[str],
    history_messages: list[Message],
    tool_name: str,
    context: str,
    correlation_id: str,
) -> AskResponse:
    """Pass 2 для search_internal: deepseek формирует ответ по контексту из Qdrant."""
    pass2_system = make_context_prompt(tool_name, context)
    pass2_messages = [
        Message(role="system", content=pass2_system),
        *history_messages,
        Message(role="user", content=masked_query),
    ]
    logger.debug(
        f"[YA STEP 11] Pass 2 (internal) → deepseek. user={masked_query!r}",
        extra={"correlation_id": correlation_id},
    )
    pass2_response = await agent.llm.chat(
        messages=pass2_messages,
        tools=None,
        correlation_id=correlation_id,
        model=Config.YANDEX_MODEL_PASS2,
    )
    masked_answer = strip_service_prefix(
        pass2_response.content or ""
    ) or "Не удалось сформировать ответ."
    final_answer = restore_pii(masked_answer, found_names)
    logger.debug(
        f"[YA STEP 12] Pass 2 (internal) ← deepseek. "
        f"answer (маскировано)={mask_text_for_logs(final_answer, found_names)!r}",
        extra={"correlation_id": correlation_id},
    )
    await save_session(
        store=agent.session_store,
        user_id=user_id,
        masked_user_msg=masked_query,
        masked_assistant_msg=masked_answer,
        correlation_id=correlation_id,
    )
    asyncio.create_task(save_analytics(
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