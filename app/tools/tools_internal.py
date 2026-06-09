"""
Реализация agent_internal tools — поиск в Qdrant с формированием контекста
для второго прохода LLM.

Три инструмента:
- search_faq        → коллекция Qdrant, фильтр source_type=faq.
                      В контекст добавляются Hidden_data / Link / Attachment.
- search_documents  → фильтр source_type ∈ {blank, regulation, support}.
- search_wiki       → фильтр source_type=wiki. ПОКА ЗАГЛУШКА (краулер не готов).

execute_internal_tool — диспетчер по имени tool. Принимает уже подключённые
qdrant_store и embedder (прокидываются из AgentLoop).

Логика выбора: какой именно tool вызвать (или ответить самой через
answer_general), LLM решает на Pass 1. Если дело дошло до tools_internal —
значит LLM сочла вопрос корпоративным и ждёт данные из наших источников.
Поэтому tools_internal всегда возвращает строку-контекст для Pass 2.

Если поиск по корпоративным данным пуст (вопрос сочтён корпоративным, но
ответа в базе нет — например, узкий вопрос, которого ещё нет в FAQ), модуль
возвращает NO_CONTEXT. На Pass 2 LLM по этому маркеру честно сообщит, что
точных данных нет, и предложит обратиться в HR через форму обратной связи.
"""
from app.core.logging import get_logger
from app.rag.embedder import Embedder
from app.rag.qdrant_store import QdrantStore, SearchResult, SourceType


logger = get_logger(__name__)


# Маппинг tool → какие source_type искать в Qdrant.
_SOURCE_TYPES: dict[str, list[SourceType]] = {
    "search_faq": ["faq"],
    "search_documents": ["blank", "regulation", "support"],
    "search_wiki": ["wiki"],
}

# Сколько ближайших чанков забираем из Qdrant под каждый tool.
_TOP_K = 5

# Маркер пустого результата по корпоративным данным. На Pass 2 LLM по этому
# маркеру скажет, что точных данных нет, и предложит форму HR.
NO_CONTEXT = "NO_CONTEXT"


async def execute_internal_tool(
    tool_name: str,
    args: dict,
    qdrant_store: QdrantStore,
    embedder: Embedder,
    correlation_id: str = "-",
) -> str:
    """
    Выполнить agent_internal tool и вернуть контекст в виде строки.

    Контекст передаётся в LLM на втором проходе (make_context_prompt).
    """
    query = (args or {}).get("query", "").strip()

    if tool_name == "search_wiki":
        # Краулер базы знаний ещё не реализован — данных по wiki пока нет.
        # Отдаём NO_CONTEXT: на Pass 2 LLM предложит обратиться в HR.
        logger.info(
            "[STUB] search_wiki called — краулер базы знаний не готов",
            extra={"correlation_id": correlation_id},
        )
        return NO_CONTEXT

    source_types = _SOURCE_TYPES.get(tool_name)
    if source_types is None:
        # Сюда попасть не должны: bot_command и answer_general не доходят до
        # tools_internal. Но если LLM вернула что-то неожиданное — не падаем.
        logger.warning(
            f"Unknown internal tool: {tool_name}",
            extra={"correlation_id": correlation_id},
        )
        return NO_CONTEXT

    if not query:
        logger.warning(
            f"{tool_name} called with empty query",
            extra={"correlation_id": correlation_id},
        )
        return NO_CONTEXT

    logger.debug(
        f"{tool_name} search: query={query!r}, source_types={source_types}",
        extra={"correlation_id": correlation_id},
    )

    # 1. Эмбеддим запрос (с префиксом query: внутри embed_query).
    query_vector = await embedder.embed_query(query, correlation_id=correlation_id)

    # 2. Ищем в Qdrant с фильтром по типу источника.
    results = await qdrant_store.search(
        query_vector=query_vector,
        top_k=_TOP_K,
        source_types=source_types,
        correlation_id=correlation_id,
    )

    if not results:
        logger.info(
            f"{tool_name}: ничего не найдено по запросу {query!r}",
            extra={"correlation_id": correlation_id},
        )
        return NO_CONTEXT

    # 3. Формируем контекст в зависимости от tool.
    if tool_name == "search_faq":
        context = _format_faq_context(results)
    else:  # search_documents
        context = _format_documents_context(results)

    logger.debug(
        f"{tool_name}: собран контекст из {len(results)} чанков, "
        f"{len(context)} символов",
        extra={"correlation_id": correlation_id},
    )
    return context


def _format_faq_context(results: list[SearchResult]) -> str:
    """
    Контекст из FAQ-результатов.

    Для каждого результата выводим текст (Вопрос/Ответ), а также Link,
    Attachment и Hidden_data, если они есть. Hidden_data содержит скрытые
    подстановки (например, «директор Иванов» вместо плейсхолдера в Answer) —
    LLM должна использовать их при формировании ответа.
    """
    blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        parts = [r.text]
        if r.hidden_data:
            parts.append(f"Скрытые данные для подстановки: {r.hidden_data}")
        if r.link:
            parts.append(f"Ссылка: {r.link}")
        if r.attachment:
            parts.append(f"Вложение: {r.attachment}")
        blocks.append(f"[Запись {i}]\n" + "\n".join(parts))
    return "\n\n".join(blocks)


def _format_documents_context(results: list[SearchResult]) -> str:
    """
    Контекст из документов (blank/regulation/support).

    source_id у документа — это его URL в CDN. Для бланков это особенно
    важно: ответ должен содержать ссылку на файл, чтобы пользователь
    мог его скачать.
    """
    blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        parts = []
        if r.title:
            parts.append(f"Документ: {r.title}")
        parts.append(r.text)
        # source_id документа — URL файла в CDN.
        if r.source_id and r.source_id.startswith("http"):
            parts.append(f"Ссылка: {r.source_id}")
        blocks.append(f"[Фрагмент {i}]\n" + "\n".join(parts))
    return "\n\n".join(blocks)