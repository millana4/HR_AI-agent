"""
Репозиторий аналитики запросов.

По каждому ТЕКСТОВОМУ ответу агента (answer_general, search_internal) пишем
строку в таблицу AI_ANALYTICS_TABLE_ID в NocoDB — для анализа HR.

Поля таблицы:
    User_hash  — обезличенный хеш сессии, один на день.
    Question   — текст вопроса, МАСКИРОВАННЫЙ, без ПД, с NAME_N.
    Date       — дата запроса (YYYY-MM-DD).
    Time       — время запроса (HH:MM:SS).
    Tool_used  — какой инструмент сформировал ответ (answer_general/search_internal).
    Answer     — маскированный текст ответа агента.

ПД в аналитику не пишем: Question и Answer берутся в маскированном виде.
"""
from datetime import datetime, timedelta, timezone

from app.core.config import Config
from app.core.logging import get_logger
from app.core.security import generate_user_hash
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)

# Московский часовой пояс (UTC+3, без перехода на летнее время).
_MSK = timezone(timedelta(hours=3))

async def save_analytics(
    nocodb_client: NocoDBClient,
    user_id: int,
    masked_question: str,
    masked_answer: str,
    tool_used: str,
    correlation_id: str = "-",
) -> None:
    """
    Записать строку аналитики в NocoDB.

    Вызывается фоново (asyncio.create_task) — не блокирует ответ пользователю.
    Любые ошибки логируются, но НЕ пробрасываются: сбой аналитики не должен
    влиять на основной ответ.

    Args:
        nocodb_client: клиент NocoDB
        user_id: Telegram ID (для вычисления User_hash, в таблицу не пишется)
        masked_question: вопрос без ПД (с NAME_N)
        masked_answer: ответ без ПД (как хранится в Redis)
        tool_used: answer_general / search_internal
        correlation_id: для трассировки
    """
    if not Config.AI_ANALYTICS_TABLE_ID:
        logger.warning(
            "AI_ANALYTICS_TABLE_ID не задан — аналитика не пишется",
            extra={"correlation_id": correlation_id},
        )
        return

    now = datetime.now(_MSK)
    record = {
        "User_hash": generate_user_hash(user_id),
        "Question": masked_question,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Tool_used": tool_used,
        "Answer": masked_answer,
    }

    try:
        await nocodb_client.create_record(
            table_id=Config.AI_ANALYTICS_TABLE_ID,
            data=record,
            correlation_id=correlation_id,
        )
        logger.debug(
            f"Аналитика записана: tool={tool_used}, "
            f"question={masked_question!r}",
            extra={"correlation_id": correlation_id},
        )
    except Exception as exc:
        # Сбой аналитики не должен влиять на ответ — только логируем.
        logger.error(
            f"Не удалось записать аналитику: {exc}",
            extra={"correlation_id": correlation_id},
        )