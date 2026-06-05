"""
CLI-точка входа для запуска индексаторов FAQ и документов.

Использование:
    python scripts/run_indexers.py --faq               # только FAQ
    python scripts/run_indexers.py --documents         # только документы
    python scripts/run_indexers.py --faq --documents   # оба (по умолчанию)
    python scripts/run_indexers.py --force             # переиндексировать всё игнорируя даты

Скрипт самодостаточен: поднимает NocoDB-клиент, Qdrant-store, embedder,
вызывает индексаторы, печатает статистику и завершает работу.
Используется как руками, так и из cron для регулярной переиндексации.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Чтобы импорты "app.*" работали при запуске скрипта напрямую
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Очистка переменных прокси ДО загрузки httpx (иначе httpx подхватит
# socks://, а он его не поддерживает — клиент Qdrant упадёт).
for _var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
             "HTTPS_PROXY", "https_proxy", "FTP_PROXY", "ftp_proxy"):
    os.environ.pop(_var, None)

from app.core.config import Config  # noqa: E402
from app.core.logging import generate_correlation_id, get_logger, setup_logging  # noqa: E402
from app.indexing.documents_indexer import index_documents  # noqa: E402
from app.indexing.faq_indexer import index_faq  # noqa: E402
from app.rag.embedder import Embedder  # noqa: E402
from app.rag.qdrant_store import QdrantStore  # noqa: E402
from app.repositories.nocodb_client import NocoDBClient  # noqa: E402


logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FAQ and/or documents indexers."
    )
    parser.add_argument(
        "--faq", action="store_true",
        help="Index FAQ table from NocoDB",
    )
    parser.add_argument(
        "--documents", action="store_true",
        help="Index documents from NocoDB+CDN",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reindex everything, ignoring indexed_at dates",
    )
    args = parser.parse_args()

    # Если не указано ни --faq ни --documents — индексируем всё
    if not args.faq and not args.documents:
        args.faq = True
        args.documents = True

    return args


async def _run(args: argparse.Namespace) -> int:
    """Прогон индексаторов. Возвращает exit code: 0 если всё ок, 1 если были ошибки."""
    correlation_id = generate_correlation_id()
    logger.info(
        f"Starting indexers: faq={args.faq}, documents={args.documents}, "
        f"force={args.force}",
        extra={"correlation_id": correlation_id},
    )

    # Поднимаем клиентов
    nocodb_client = NocoDBClient()
    qdrant_store = QdrantStore()
    embedder = Embedder()

    try:
        await qdrant_store.connect()
        await qdrant_store.ensure_collection(correlation_id=correlation_id)
    except Exception as exc:
        logger.exception(
            f"Failed to connect to Qdrant: {exc}",
            extra={"correlation_id": correlation_id},
        )
        return 1

    total_errors = 0

    try:
        if args.faq:
            faq_stats = await index_faq(
                nocodb_client, qdrant_store, embedder,
                force=args.force, correlation_id=correlation_id,
            )
            print(f"FAQ indexer: {faq_stats}")
            total_errors += faq_stats.errors

        if args.documents:
            doc_stats = await index_documents(
                nocodb_client, qdrant_store, embedder,
                force=args.force, correlation_id=correlation_id,
            )
            print(f"Documents indexer: {doc_stats}")
            total_errors += doc_stats.errors
    finally:
        await qdrant_store.disconnect()

    if total_errors > 0:
        logger.warning(
            f"Indexers finished with {total_errors} errors",
            extra={"correlation_id": correlation_id},
        )
        return 1

    logger.info(
        "Indexers finished successfully",
        extra={"correlation_id": correlation_id},
    )
    return 0


def main() -> None:
    setup_logging()
    args = _parse_args()
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()