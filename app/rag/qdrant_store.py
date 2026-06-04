"""
Обёртка над Qdrant — векторной базой данных.

Хранит чанки текста с эмбеддингами и метаданными.
Метаданные содержат source_type, source_url, title и сам текст чанка
(чтобы при поиске сразу получать контекст без обращения к источнику).
"""
import uuid
from dataclasses import dataclass
from typing import Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.core.config import Config
from app.core.exceptions import RepositoryError
from app.core.logging import get_logger
from app.rag.embedder import VECTOR_SIZE


logger = get_logger(__name__)


SourceType = Literal[
    "blank",        # бланки документов (только метаданные, не содержимое)
    "regulation",   # нормативные документы из CDN
    "support",      # справочные документы из CDN (велкомбук, гайды)
    "wiki_page",    # страницы внешней базы знаний
    "faq",          # пары вопрос-ответ из таблицы AI_FAQ
]


@dataclass
class QdrantChunk:
    """Чанк для записи в Qdrant — вектор + метаданные."""

    vector: list[float]
    source_type: SourceType
    source_url: str
    title: str
    text: str
    chunk_index: int = 0
    indexed_at: str | None = None  # ISO-строка времени индексации


@dataclass
class SearchResult:
    """Результат поиска в Qdrant — чанк + score."""

    source_type: str
    source_url: str
    title: str
    text: str
    chunk_index: int
    score: float


class QdrantStore:
    """
    Обёртка над AsyncQdrantClient.

    Использование:
        store = QdrantStore()
        await store.connect()
        await store.ensure_collection()
        await store.upsert([chunk1, chunk2])
        results = await store.search(query_vector, top_k=5)
        await store.disconnect()
    """

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None
        self._collection = Config.QDRANT_COLLECTION

    async def connect(self) -> None:
        """Подключиться к Qdrant."""
        self._client = AsyncQdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT,
        )
        try:
            await self._client.get_collections()
        except Exception as exc:
            raise RepositoryError(f"Failed to connect to Qdrant: {exc}") from exc

        logger.info(
            f"Connected to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}"
        )

    async def disconnect(self) -> None:
        """Закрыть соединение."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RepositoryError("QdrantStore is not connected. Call connect() first.")
        return self._client

    async def ensure_collection(self, correlation_id: str = "-") -> None:
        """Создать коллекцию, если её ещё нет. Idempotent."""
        client = self._get_client()
        try:
            existing = await client.get_collections()
            names = {c.name for c in existing.collections}
        except Exception as exc:
            raise RepositoryError(f"Qdrant get_collections failed: {exc}") from exc

        if self._collection in names:
            logger.debug(
                f"Qdrant collection '{self._collection}' already exists",
                extra={"correlation_id": correlation_id},
            )
            return

        try:
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qm.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise RepositoryError(f"Qdrant create_collection failed: {exc}") from exc

        logger.info(
            f"Created Qdrant collection '{self._collection}'",
            extra={"correlation_id": correlation_id},
        )

    async def upsert(
        self,
        chunks: list[QdrantChunk],
        correlation_id: str = "-",
    ) -> None:
        """
        Записать чанки в коллекцию.

        Если чанк с такими же source_url + chunk_index уже есть — перезаписывает.
        Это позволяет безопасно переиндексировать один и тот же документ.
        """
        if not chunks:
            return

        client = self._get_client()
        points = [
            qm.PointStruct(
                id=_make_point_id(chunk.source_url, chunk.chunk_index),
                vector=chunk.vector,
                payload={
                    "source_type": chunk.source_type,
                    "source_url": chunk.source_url,
                    "title": chunk.title,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "indexed_at": chunk.indexed_at,
                },
            )
            for chunk in chunks
        ]

        try:
            await client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise RepositoryError(f"Qdrant upsert failed: {exc}") from exc

        logger.info(
            f"Upserted {len(points)} chunks to Qdrant",
            extra={"correlation_id": correlation_id},
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_types: list[SourceType] | None = None,
        correlation_id: str = "-",
    ) -> list[SearchResult]:
        """
        Найти top_k ближайших чанков.

        Args:
            query_vector: эмбеддинг запроса
            top_k: количество результатов
            source_types: фильтр по типу источника (например, только ['regulation', 'support'])
            correlation_id: для логов

        Returns:
            Список SearchResult, отсортированных по убыванию score.
        """
        client = self._get_client()

        query_filter = None
        if source_types:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="source_type",
                        match=qm.MatchAny(any=list(source_types)),
                    )
                ]
            )

        try:
            response = await client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise RepositoryError(f"Qdrant search failed: {exc}") from exc

        results = [
            SearchResult(
                source_type=point.payload.get("source_type", ""),
                source_url=point.payload.get("source_url", ""),
                title=point.payload.get("title", ""),
                text=point.payload.get("text", ""),
                chunk_index=point.payload.get("chunk_index", 0),
                score=point.score,
            )
            for point in response.points
        ]

        logger.debug(
            f"Qdrant search returned {len(results)} results",
            extra={"correlation_id": correlation_id},
        )
        return results

    async def delete_by_source(
        self,
        source_url: str,
        correlation_id: str = "-",
    ) -> None:
        """Удалить все чанки одного источника. Используется при переиндексации."""
        client = self._get_client()
        try:
            await client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="source_url",
                                match=qm.MatchValue(value=source_url),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            raise RepositoryError(f"Qdrant delete failed: {exc}") from exc

        logger.info(
            f"Deleted chunks for source: {source_url}",
            extra={"correlation_id": correlation_id},
        )

    async def get_indexed_at(
            self,
            source_url: str,
            correlation_id: str = "-",
    ) -> str | None:
        """
        Получить дату последней индексации источника.

        Возвращает значение payload.indexed_at у любого чанка с указанным source_url,
        или None если такого источника в коллекции ещё нет.

        Дата хранится в ISO-строке.
        """
        client = self._get_client()
        try:
            response = await client.scroll(
                collection_name=self._collection,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source_url",
                            match=qm.MatchValue(value=source_url),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise RepositoryError(f"Qdrant scroll failed: {exc}") from exc

        points, _next_offset = response
        if not points:
            return None

        return points[0].payload.get("indexed_at")


def _make_point_id(source_url: str, chunk_index: int) -> str:
    """
    Сгенерировать стабильный UUID для точки Qdrant.

    Один и тот же (source_url, chunk_index) → один и тот же UUID.
    Это позволяет переиндексировать тот же документ без дублирования.
    """
    raw = f"{source_url}::chunk::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))