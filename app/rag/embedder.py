"""
Embedder на основе multilingual-e5-large.

Модель загружается лениво при первом вызове (~2 ГБ).
По умолчанию кешируется в ~/.cache/huggingface/.

У моделей семейства e5 есть требование: добавлять префикс к тексту перед эмбеддингом.
- Для документов в индексе:  "passage: <текст>"
- Для поисковых запросов:    "query: <текст>"

Без префиксов качество поиска заметно падает.
"""
import asyncio
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger


logger = get_logger(__name__)


MODEL_NAME = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024  # размерность multilingual-e5-large

# Префиксы согласно документации e5
_DOCUMENT_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


class Embedder:
    """
    Обёртка над sentence-transformers для расчёта эмбеддингов.

    Singleton-паттерн: на весь сервис достаточно одного экземпляра модели.
    Используй get_embedder() для получения общего инстанса.
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _ensure_loaded(self) -> SentenceTransformer:
        """Загрузить модель при первом обращении."""
        if self._model is None:
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            self._model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded")
        return self._model

    async def embed_documents(
        self,
        texts: Sequence[str],
        correlation_id: str = "-",
    ) -> list[list[float]]:
        """
        Эмбеддит тексты как документы (для индексации).

        Args:
            texts: список текстов
            correlation_id: для логов

        Returns:
            Список векторов размерности VECTOR_SIZE.
        """
        if not texts:
            return []

        prefixed = [_DOCUMENT_PREFIX + t for t in texts]
        logger.debug(
            f"Embedding {len(texts)} documents",
            extra={"correlation_id": correlation_id},
        )
        return await self._encode(prefixed)

    async def embed_query(
        self,
        text: str,
        correlation_id: str = "-",
    ) -> list[float]:
        """
        Эмбеддит поисковый запрос.

        Args:
            text: текст запроса
            correlation_id: для логов

        Returns:
            Вектор размерности VECTOR_SIZE.
        """
        logger.debug(
            f"Embedding query: {text[:60]!r}",
            extra={"correlation_id": correlation_id},
        )
        vectors = await self._encode([_QUERY_PREFIX + text])
        return vectors[0]

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        """Сам расчёт эмбеддингов — синхронный, гонится в отдельном thread."""
        model = self._ensure_loaded()

        def run_encode() -> list[list[float]]:
            # normalize_embeddings=True — нормализует векторы к длине 1,
            # что важно для корректной работы косинусного расстояния в Qdrant
            np_array = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np_array.tolist()

        return await asyncio.to_thread(run_encode)


# Singleton
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Получить глобальный экземпляр Embedder."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder