"""
Общие хелперы для петель обработки запросов (GigaChat и Yandex).

Вынесено из agent_loop.py, чтобы обе петли (agent_loop.py для GigaChat и
agent_loop_yandex.py для Yandex) использовали один и тот же код для:
- маскирования ПД в логах,
- восстановления реальных имён из плейсхолдеров NAME_N,
- сохранения пары сообщений в Redis-сессию,
- очистки служебных префиксов из ответа LLM.

PII-политика (едина для обеих петель):
- В Redis и аналитику уходят только МАСКИРОВАННЫЕ версии (NAME_1, NAME_2, ...).
- Восстановление NAME_N → реальные имена — только в том, что уходит наружу
  (финальный ответ, args для бота). В лог реальные имена не пишем.
"""
import re

from app.core.logging import get_logger
from app.services.pii_parser import make_placeholder, mask_for_logs
from app.services.session_store import SessionStore


logger = get_logger(__name__)


# Префиксы, которые LLM иногда добавляет в начало ответа из промпта
# (например, "Tool: search_internal", "answer_general:"). В ответ пользователю
# они попадать не должны — срезаем.
_SERVICE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:tool\s*:\s*\w+|answer_general)\s*:?\s*\n?",
    re.IGNORECASE,
)


def strip_service_prefix(text: str) -> str:
    """Убрать служебный префикс вида 'Tool: search_internal' из начала ответа."""
    cleaned = _SERVICE_PREFIX_PATTERN.sub("", text, count=1)
    return cleaned.strip()


def restore_pii(text: str, original_names: list[str]) -> str:
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


def restore_pii_in_args(args: dict, original_names: list[str]) -> dict:
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
            restored[key] = restore_pii(value, original_names)
        else:
            restored[key] = value
    return restored


def mask_text_for_logs(text: str, original_names: list[str]) -> str:
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


def mask_args_for_logs(args: dict, original_names: list[str]) -> dict:
    """Маскировать строковые значения args для лога."""
    if not original_names or not args:
        return args
    masked: dict = {}
    for key, value in args.items():
        if isinstance(value, str):
            masked[key] = mask_text_for_logs(value, original_names)
        else:
            masked[key] = value
    return masked


async def save_session(
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