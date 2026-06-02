import logging
import sys
from logging.handlers import RotatingFileHandler
from uuid import uuid4

from app.core.config import Config, PROJECT_ROOT

"""
Настройка логирования с явной передачей correlation_id.
correlation_id пробрасывается параметром функций по всей цепочке вызовов.

Использование:
    from app.core.logging import setup_logging, get_logger, generate_correlation_id

    setup_logging()  # один раз при старте приложения

    logger = get_logger(__name__)
    cid = generate_correlation_id()
    logger.info("Message", extra={"correlation_id": cid})
"""

def generate_correlation_id() -> str:
    """Сгенерировать новый correlation_id для нового запроса."""
    return uuid4().hex[:16]


class CorrelationIdFilter(logging.Filter):
    """
    Гарантирует, что у каждой записи лога есть атрибут correlation_id.

    Если correlation_id не передан через extra={...} — подставляет прочерк.
    Это нужно, чтобы формат логов не падал на записях без correlation_id.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = "-"
        return True


def setup_logging() -> None:
    """Настроить логирование для всего приложения. Вызывается один раз при старте."""
    log_file_path = PROJECT_ROOT / Config.LOG_FILE
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    correlation_filter = CorrelationIdFilter()

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=Config.LOG_MAX_BYTES,
        backupCount=Config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(correlation_filter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(correlation_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Снижаем уровень для шумных библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Получить именованный логгер. Используй __name__ в качестве имени."""
    return logging.getLogger(name)