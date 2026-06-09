"""
Обёртка над Qdrant — векторной базой данных.

Хранит чанки текста с эмбеддингами и метаданными.
Метаданные содержат source_type, source_id, title и сам текст чанка
(чтобы при поиске сразу получать контекст без обращения к источнику).

source_id — универсальный идентификатор источника:
- для документов из CDN это URL файла,
- для FAQ из NocoDB — строка "nocodb_id:<Id>",
- для страниц wiki — URL страницы.
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
    "blank",
    "regulation",
    "support",
    "wiki",
    "faq",
]


@dataclass
class QdrantChunk:
    """Чанк для записи в Qdrant — вектор + метаданные."""

    vector: list[float]
    source_type: SourceType
    source_id: str
    title: str
    text: str
    chunk_index: int = 0
    indexed_at: str | None = None  # ISO-строка времени индексации
    # Доп. поля FAQ — не эмбеддятся, нужны при формировании ответа.
    link: str | None = None
    attachment: str | None = None
    hidden_data: str | None = None


@dataclass
class SearchResult:
    """Результат поиска в Qdrant — чанк + score."""

    source_type: str
    source_id: str
    title: str
    text: str
    chunk_index: int
    score: float
    # Доп. поля FAQ (могут отсутствовать у документов/wiki).
    link: str | None = None
    attachment: str | None = None
    hidden_data: str | None = None


class QdrantStore:
    """
    Обёртка над AsyncQdrantClient.
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

        Если чанк с такими же source_id + chunk_index уже есть — перезаписывает.
        """
        if not chunks:
            return

        client = self._get_client()
        points = [
            qm.PointStruct(
                id=_make_point_id(chunk.source_id, chunk.chunk_index),
                vector=chunk.vector,
                payload={
                    "source_type": chunk.source_type,
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "indexed_at": chunk.indexed_at,
                    "link": chunk.link,
                    "attachment": chunk.attachment,
                    "hidden_data": chunk.hidden_data,
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
        """Найти top_k ближайших чанков."""
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
                source_id=point.payload.get("source_id", ""),
                title=point.payload.get("title", ""),
                text=point.payload.get("text", ""),
                chunk_index=point.payload.get("chunk_index", 0),
                score=point.score,
                link=point.payload.get("link"),
                attachment=point.payload.get("attachment"),
                hidden_data=point.payload.get("hidden_data"),
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
        source_id: str,
        correlation_id: str = "-",
    ) -> None:
        """Удалить все чанки одного источника."""
        client = self._get_client()
        try:
            await client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="source_id",
                                match=qm.MatchValue(value=source_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            raise RepositoryError(f"Qdrant delete failed: {exc}") from exc

        logger.info(
            f"Deleted chunks for source: {source_id}",
            extra={"correlation_id": correlation_id},
        )

    async def get_indexed_at(
        self,
        source_id: str,
        correlation_id: str = "-",
    ) -> str | None:
        """
        Получить дату последней индексации источника.

        Возвращает значение payload.indexed_at у любого чанка с указанным source_id,
        или None если такого источника в коллекции ещё нет.
        """
        client = self._get_client()
        try:
            response = await client.scroll(
                collection_name=self._collection,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source_id",
                            match=qm.MatchValue(value=source_id),
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


def _make_point_id(source_id: str, chunk_index: int) -> str:
    """
    Сгенерировать стабильный UUID для точки Qdrant.

    Один и тот же (source_id, chunk_index) → один и тот же UUID.
    """
    raw = f"{source_id}::chunk::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))