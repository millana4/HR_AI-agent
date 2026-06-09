"""
Реализация agent_internal tool — поиск во всех внутренних источниках.

Один инструмент:
- search_internal → ищет в Qdrant сразу по всем внутренним источникам:
  FAQ, документы (blank/regulation/support) и база знаний (wiki).
  Ответ может лежать в любом из них, поэтому ищем везде одним запросом.

execute_internal_tool принимает уже подключённые qdrant_store и embedder
(прокидываются из AgentLoop).

Если поиск вернул 0 результатов — отдаём NO_CONTEXT. agent_loop по этому
маркеру предложит пользователю форму обращения в HR.
"""
from app.core.logging import get_logger
from app.rag.embedder import Embedder
from app.rag.qdrant_store import QdrantStore, SearchResult, SourceType


logger = get_logger(__name__)


# search_internal ищет по всем внутренним источникам сразу.
_INTERNAL_SOURCES: list[SourceType] = [
    "faq",
    "blank",
    "regulation",
    "support",
    "wiki",
]

# Сколько ближайших чанков забираем из Qdrant.
_TOP_K = 5

# Маркер пустого результата. agent_loop по нему предложит форму HR.
NO_CONTEXT = "NO_CONTEXT"


async def execute_internal_tool(
    tool_name: str,
    args: dict,
    qdrant_store: QdrantStore,
    embedder: Embedder,
    correlation_id: str = "-",
) -> str:
    """
    Выполнить search_internal и вернуть контекст в виде строки.

    Контекст передаётся в LLM на втором проходе (make_context_prompt).
    Если ничего не нашлось — возвращает NO_CONTEXT.
    """
    if tool_name != "search_internal":
        # Сюда попасть не должны: bot_command и answer_general не доходят
        # до tools_internal. Но если LLM вернула неожиданное — не падаем.
        logger.warning(
            f"Unknown internal tool: {tool_name}",
            extra={"correlation_id": correlation_id},
        )
        return NO_CONTEXT

    query = (args or {}).get("query", "").strip()
    if not query:
        logger.warning(
            "search_internal called with empty query",
            extra={"correlation_id": correlation_id},
        )
        return NO_CONTEXT

    logger.debug(
        f"search_internal: query={query!r}, source_types={_INTERNAL_SOURCES}",
        extra={"correlation_id": correlation_id},
    )

    # 1. Эмбеддим запрос (с префиксом query: внутри embed_query).
    query_vector = await embedder.embed_query(query, correlation_id=correlation_id)

    # 2. Ищем в Qdrant по всем внутренним источникам сразу.
    results = await qdrant_store.search(
        query_vector=query_vector,
        top_k=_TOP_K,
        source_types=_INTERNAL_SOURCES,
        correlation_id=correlation_id,
    )

    if not results:
        logger.info(
            f"search_internal: ничего не найдено по запросу {query!r}",
            extra={"correlation_id": correlation_id},
        )
        return NO_CONTEXT

    # 3. Формируем смешанный контекст (FAQ + документы + wiki).
    context = _format_context(results)

    logger.debug(
        f"search_internal: собран контекст из {len(results)} чанков, "
        f"{len(context)} символов",
        extra={"correlation_id": correlation_id},
    )
    return context


def _format_context(results: list[SearchResult]) -> str:
    """
    Контекст из смешанных результатов (FAQ + документы + wiki).

    Каждый чанк форматируется по своему source_type:
    - faq → текст + Hidden_data / Link / Attachment;
    - blank / regulation / support → заголовок + текст + URL файла в CDN;
    - wiki → заголовок + текст + URL страницы.
    """
    blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        if r.source_type == "faq":
            parts = [r.text]
            if r.hidden_data:
                parts.append(f"Скрытые данные для подстановки: {r.hidden_data}")
            if r.link:
                parts.append(f"Ссылка: {r.link}")
            if r.attachment:
                parts.append(f"Вложение: {r.attachment}")
            blocks.append(f"[Запись {i}]\n" + "\n".join(parts))
        else:
            # Документы (blank/regulation/support) и wiki.
            parts = []
            if r.title:
                parts.append(f"Документ: {r.title}")
            parts.append(r.text)
            # source_id у документа/wiki — это URL (файла в CDN или страницы).
            if r.source_id and r.source_id.startswith("http"):
                parts.append(f"Ссылка: {r.source_id}")
            blocks.append(f"[Фрагмент {i}]\n" + "\n".join(parts))
    return "\n\n".join(blocks)