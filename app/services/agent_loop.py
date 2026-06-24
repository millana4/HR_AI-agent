"""
Главный мозг агента — двухпроходная схема обработки запроса.

Pass 1 (выбор действия): LLM получает текущий запрос + список tools (без истории).
Решает одно из трёх:
  (a) ответить сама текстом              → answer_general
  (b) вызвать search_internal            → Pass 2 (поиск во внутренних источниках)
  (c) вызвать bot_command (контакты и пр.) → возврат боту команды

Pass 2 (формирование ответа): выполняем search_internal, отдаём найденный
контекст + историю в LLM, получаем финальный текст. Если поиск пуст —
возвращаем боту suggest_hr_form.

История диалога:
  - На Pass 1 НЕ подмешивается (выбор tool — только по текущему вопросу).
  - На Pass 2 подмешивается как вспомогательный контекст.
  - bot_command и пустой поиск в историю НЕ пишутся (агент по ним не формирует
    текстовый ответ — иначе копятся непарные user-реплики и путают LLM).

Аналитика:
  - По каждому ТЕКСТОВОМУ ответу (answer_general, search_internal) фоново
    пишем строку в NocoDB (asyncio.create_task) — не блокируя ответ.
  - bot_command в аналитику не пишем (нет текстового ответа агента).

PII-политика:
  - В Redis и в аналитику сохраняем МАСКИРОВАННЫЕ версии (NAME_1, NAME_2, ...).
  - Восстановление NAME_N → реальные имена делается только в том, что уходит
    наружу (финальный ответ, args для бота). В лог реальные имена не пишем —
    только маскированные (mask_for_logs: 2 буквы + звёздочки).

Нумерация шагов в логах [STEP N] соответствует комментариям ниже.
"""
import re
import asyncio
from dataclasses import dataclass

from app.api.schemas import AskResponse, TextResponse, ToolCall, ToolCallResponse
from app.core.config import Config
from app.core.logging import get_logger
from app.repositories.analytics import save_analytics
from app.llm.base import BaseLLMClient, Message
from app.llm.prompts_gigachat import (
    make_context_prompt,
    make_system_prompt_with_departments,
)
from app.rag.embedder import Embedder
from app.rag.qdrant_store import QdrantStore
from app.repositories.nocodb_client import NocoDBClient
from app.services.address_cache import AddressCache
from app.services.departments_cache import DepartmentsCache
from app.services.pii_parser import PiiParser, make_placeholder, mask_for_logs
from app.services.session_store import SessionStore
from app.tools.registry import get_all_tool_specs, is_agent_internal, is_bot_command
from app.tools.tools_internal import NO_CONTEXT, execute_internal_tool


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
    qdrant_store: QdrantStore
    embedder: Embedder
    departments_cache: DepartmentsCache
    address_cache: AddressCache
    fallback_llm: BaseLLMClient | None = None


def _mask_text_for_logs(text: str, original_names: list[str]) -> str:
    """
    Подготовить текст для лога: реальные имена → маскированные (2 буквы + ***).

    Применяется к строкам, где реальные ПД уже восстановлены (финальный ответ,
    args для бота). В сам лог реальные имена не попадают.
    """
    if not original_names:
        return text
    result = text
    for name in original_names:
        result = result.replace(name, mask_for_logs(name))
    return result


def _mask_args_for_logs(args: dict, original_names: list[str]) -> dict:
    """Маскировать строковые значения args для лога."""
    if not original_names or not args:
        return args
    masked: dict = {}
    for key, value in args.items():
        if isinstance(value, str):
            masked[key] = _mask_text_for_logs(value, original_names)
        else:
            masked[key] = value
    return masked


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

    # ========================================================================
    # ШАГ 1. Маскируем ПД (фамилии/имена) в запросе: реальное имя → NAME_N.
    # ========================================================================
    pii_result = agent.pii_parser.parse(request_text, correlation_id=correlation_id)
    masked_query = pii_result.masked_text
    found_names = pii_result.found_names
    logger.debug(
        f"[STEP 1] PII-маскирование: masked_query={masked_query!r}, "
        f"найдено имён (маскировано)={[mask_for_logs(n) for n in found_names]}",
        extra={"correlation_id": correlation_id},
    )

    # ========================================================================
    # ШАГ 2. Загружаем историю диалога из Redis (она хранится маскированной).
    #        Понадобится только на Pass 2.
    # ========================================================================
    history_dicts = await agent.session_store.get_history(
        user_id=user_id, correlation_id=correlation_id
    )
    history_messages = [
        Message(role=d["role"], content=d["content"]) for d in history_dicts
    ]
    logger.debug(
        f"[STEP 2] История из Redis: {len(history_messages)} сообщений: "
        f"{[(m.role, m.content) for m in history_messages]}",
        extra={"correlation_id": correlation_id},
    )

    # ========================================================================
    # ШАГ 3. Получаем списки отделов (Мавис/Вотоня) из кеша Redis и подставляем
    #        их в системный промпт — чтобы LLM выбирала отдел из реального
    #        справочника, а не угадывала.
    # ========================================================================
    mavis_deps, votonia_deps = await agent.departments_cache.get_departments(
        agent.nocodb_client, correlation_id=correlation_id
    )
    logger.debug(
        f"[STEP 3] Отделы из кеша: Мавис ({len(mavis_deps)})={mavis_deps}, "
        f"Вотоня ({len(votonia_deps)})={votonia_deps}",
        extra={"correlation_id": correlation_id},
    )
    system_prompt = make_system_prompt_with_departments(mavis_deps, votonia_deps)

    # ========================================================================
    # ШАГ 4. Pass 1 — выбор действия. Историю НЕ подмешиваем: выбор tool
    #        должен опираться только на текущий вопрос.
    # ========================================================================
    pass1_messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=masked_query),
    ]
    logger.debug(
        f"[STEP 4] Pass 1 → LLM (без истории). "
        f"messages={[(m.role, m.content) for m in pass1_messages]}",
        extra={"correlation_id": correlation_id},
    )
    llm_response = await agent.llm.chat(
        messages=pass1_messages,
        tools=get_all_tool_specs(),
        correlation_id=correlation_id,
        model=Config.YANDEX_MODEL_PASS1,
    )
    logger.debug(
        f"[STEP 4] Pass 1 ← LLM: type={llm_response.type}, "
        f"content={llm_response.content!r}, "
        f"tool_calls={[(tc.name, tc.args) for tc in llm_response.tool_calls]}",
        extra={"correlation_id": correlation_id},
    )

    # ========================================================================
    # ШАГ 5. Ветка (a): LLM ответила текстом → answer_general.
    #        Восстанавливаем ПД, сохраняем пару в историю, пишем аналитику,
    #        отдаём ответ.
    # ========================================================================
    if llm_response.type == "text":
        masked_answer = _strip_service_prefix(llm_response.content or "")
        final_answer = _restore_pii(masked_answer, found_names)
        logger.debug(
            f"[STEP 5] answer_general. masked_answer={masked_answer!r} → "
            f"final_answer (маскировано)="
            f"{_mask_text_for_logs(final_answer, found_names)!r}",
            extra={"correlation_id": correlation_id},
        )
        await _save_session(
            store=agent.session_store,
            user_id=user_id,
            masked_user_msg=masked_query,
            masked_assistant_msg=masked_answer,
            correlation_id=correlation_id,
        )
        # Аналитика — фоново, маскированные версии (без ПД).
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

    # ========================================================================
    # ШАГ 6. LLM вызвала tool. Берём первый (параллельные вызовы не поддерживаем).
    # ========================================================================
    tool_call = llm_response.tool_calls[0]
    tool_name = tool_call.name
    tool_args = tool_call.args
    logger.info(
        f"[STEP 6] LLM выбрала tool: {tool_name}",
        extra={"correlation_id": correlation_id},
    )

    # ========================================================================
    # ШАГ 7. Ветка (c): bot_command (контакты, телефоны, форма HR).
    #        Возвращаем боту команду. Восстанавливаем ПД в args.
    #        В историю и аналитику НЕ пишем (агент не формирует текстовый ответ).
    # ========================================================================
    if is_bot_command(tool_name):
        restored_args = _restore_pii_in_args(tool_args, found_names)
        logger.debug(
            f"[STEP 7] bot_command {tool_name}: args от LLM={tool_args} → "
            f"restored_args (маскировано)="
            f"{_mask_args_for_logs(restored_args, found_names)}",
            extra={"correlation_id": correlation_id},
        )
        # Аналитика — фоново. Для bot_command текст ответа формирует бот,
        # поэтому пишем только вопрос и название tool, Answer пустой.
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

    # ========================================================================
    # ШАГ 8. Защита: LLM вызвала неизвестный/неподдерживаемый tool.
    #        Fallback на вежливое сообщение.
    # ========================================================================
    if not is_agent_internal(tool_name):
        logger.warning(
            f"[STEP 8] Неизвестный tool: {tool_name} — fallback",
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

    # ========================================================================
    # ШАГ 9. Ветка (b): search_internal — ищем во внутренних источниках Qdrant.
    # ========================================================================
    context = await execute_internal_tool(
        tool_name=tool_name,
        args=tool_args,
        qdrant_store=agent.qdrant_store,
        embedder=agent.embedder,
        correlation_id=correlation_id,
    )
    logger.debug(
        f"[STEP 9] search_internal вернул контекст ({len(context)} симв.): "
        f"{context!r}",
        extra={"correlation_id": correlation_id},
    )

    # ========================================================================
    # ШАГ 10. Поиск пуст → предлагаем боту показать форму HR.
    #         В историю и аналитику НЕ пишем (нет текстового ответа агента).
    # ========================================================================
    if context == NO_CONTEXT:
        logger.info(
            "[STEP 10] search_internal вернул пусто — предлагаем форму HR",
            extra={"correlation_id": correlation_id},
        )
        # Аналитика — фоново. Текст ответа отсутствует (форму показывает бот),
        # поэтому Answer пустой, Tool_used = suggest_hr_form.
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

    # ========================================================================
    # ШАГ 11. Pass 2 — LLM формирует финальный ответ из найденного контекста.
    #         Подмешиваем историю как вспомогательный фон. GigaChat требует
    #         один system-промпт в начале, поэтому контекст склеиваем с ним.
    #         Отделы здесь не нужны (tool уже выбран) — передаём пустой блок.
    # ========================================================================
    pass2_system = (
        make_system_prompt_with_departments([], [])
        + "\n\n"
        + make_context_prompt(tool_name, context)
    )
    pass2_messages = [
        Message(role="system", content=pass2_system),
        *history_messages,
        Message(role="user", content=masked_query),
    ]
    logger.debug(
        f"[STEP 11] Pass 2 → LLM. "
        f"messages={[(m.role, m.content) for m in pass2_messages]}",
        extra={"correlation_id": correlation_id},
    )
    pass2_response = await agent.llm.chat(
        messages=pass2_messages,
        tools=None,  # на втором проходе tools не нужны
        correlation_id=correlation_id,
        model=Config.YANDEX_MODEL_PASS2,
    )

    # ========================================================================
    # ШАГ 12. Восстанавливаем ПД в финальном тексте, сохраняем пару в историю,
    #         пишем аналитику, отдаём ответ.
    # ========================================================================
    masked_answer = _strip_service_prefix(
        pass2_response.content or ""
    ) or "Не удалось сформировать ответ."
    final_answer = _restore_pii(masked_answer, found_names)
    logger.debug(
        f"[STEP 12] Pass 2 ← LLM. masked_answer={masked_answer!r} → "
        f"final_answer (маскировано)="
        f"{_mask_text_for_logs(final_answer, found_names)!r}",
        extra={"correlation_id": correlation_id},
    )
    await _save_session(
        store=agent.session_store,
        user_id=user_id,
        masked_user_msg=masked_query,
        masked_assistant_msg=masked_answer,
        correlation_id=correlation_id,
    )
    # Аналитика — фоново, маскированные версии (без ПД).
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


def _restore_pii_in_args(args: dict, original_names: list[str]) -> dict:
    """
    Восстановить реальные имена в значениях args для bot_command.

    LLM возвращает аргументы с плейсхолдерами NAME_1, NAME_2 (например,
    {"query": "NAME_1"}). Боту нужны реальные фамилии/имена для поиска
    по справочникам. Подставляем по номеру: NAME_1 → original_names[0].

    Затрагиваем только строковые значения.
    """
    if not original_names or not args:
        return args

    restored: dict = {}
    for key, value in args.items():
        if isinstance(value, str):
            restored[key] = _restore_pii(value, original_names)
        else:
            restored[key] = value
    return restored


# Префиксы, которые LLM иногда добавляет в начало ответа из промпта
# (например, "Tool: search_internal", "answer_general:"). В ответ пользователю
# они попадать не должны — срезаем.
_SERVICE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:tool\s*:\s*\w+|answer_general)\s*:?\s*\n?",
    re.IGNORECASE,
)
def _strip_service_prefix(text: str) -> str:
    """Убрать служебный префикс вида 'Tool: search_internal' из начала ответа."""
    cleaned = _SERVICE_PREFIX_PATTERN.sub("", text, count=1)
    return cleaned.strip()


def _restore_pii(text: str, original_names: list[str]) -> str:
    """
    Заменить плейсхолдеры NAME_1, NAME_2, ... на реальные имена по номеру.

    NAME_1 → original_names[0], NAME_2 → original_names[1] и т.д.
    Если для плейсхолдера нет соответствующего имени — оставляем как есть.
    """
    if not original_names:
        return text
    result = text
    for i, name in enumerate(original_names, start=1):
        placeholder = make_placeholder(i)
        result = result.replace(placeholder, name)
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
    не должно. Восстановление ПД происходит только в финальном ответе
    пользователю.
    """
    logger.debug(
        f"[STEP 13] Запись в Redis (маскированное): "
        f"user={masked_user_msg!r}, assistant={masked_assistant_msg!r}",
        extra={"correlation_id": correlation_id},
    )
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