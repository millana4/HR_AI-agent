"""
Реализация agent_internal tools.

Сейчас здесь заглушки — в шаге 13 заменим на реальный поиск в Qdrant
с PII-обработкой Hidden_data и фильтрами по source_type.

execute_internal_tool — диспетчер по имени tool.
"""
from app.core.logging import get_logger


logger = get_logger(__name__)


async def execute_internal_tool(
    tool_name: str,
    args: dict,
    correlation_id: str = "-",
) -> str:
    """
    Выполнить agent_internal tool и вернуть контекст в виде строки.

    Контекст потом передаётся в LLM на втором проходе.
    """
    query = args.get("query", "")

    if tool_name == "search_faq":
        return await _search_faq(query, correlation_id)
    if tool_name == "search_documents":
        return await _search_documents(query, correlation_id)
    if tool_name == "search_wiki":
        return await _search_wiki(query, correlation_id)

    logger.warning(
        f"Unknown internal tool: {tool_name}",
        extra={"correlation_id": correlation_id},
    )
    return ""


async def _search_faq(query: str, correlation_id: str) -> str:
    """Заглушка для search_faq. В шаге 13 заменим на поиск в Qdrant."""
    logger.debug(
        f"[STUB] search_faq called with query: {query!r}",
        extra={"correlation_id": correlation_id},
    )
    return f"[заглушка search_faq для запроса: {query}]"


async def _search_documents(query: str, correlation_id: str) -> str:
    """Заглушка для search_documents."""
    logger.debug(
        f"[STUB] search_documents called with query: {query!r}",
        extra={"correlation_id": correlation_id},
    )
    return f"[заглушка search_documents для запроса: {query}]"


async def _search_wiki(query: str, correlation_id: str) -> str:
    """Заглушка для search_wiki."""
    logger.debug(
        f"[STUB] search_wiki called with query: {query!r}",
        extra={"correlation_id": correlation_id},
    )
    return f"[заглушка search_wiki для запроса: {query}]"