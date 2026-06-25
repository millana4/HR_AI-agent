"""
Утилиты безопасности: маскирование PII, хеширование пользовательских сессий.
"""
import hashlib
import re

from app.core.config import Config


_PHONE_PATTERN = re.compile(
    r"(?:\+?7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def mask_pii(text: str) -> str:
    """
    Маскирует телефоны и email в тексте.

    Используется для логирования и записи в аналитическую таблицу.

    Имена и фамилии маскируются отдельно через PII-парсер на основе Natasha
    (см. app/services/pii_parser.py) — здесь только базовая логика.
    """
    if not text:
        return text

    masked = _PHONE_PATTERN.sub("[PHONE]", text)
    masked = _EMAIL_PATTERN.sub("[EMAIL]", masked)
    return masked


def generate_user_hash(user_id: int) -> str:
    """
    Генерирует обезличенный хеш пользователя для записи в аналитику.

    Хеш зависит от (user_id, salt) и НЕ зависит от даты — поэтому один
    пользователь всегда получает один и тот же хеш, в любой день. Это
    позволяет отслеживать активность пользователя во времени, не раскрывая
    его Telegram ID.

    Соль (SESSION_HASH_SALT) защищает от перебора: восстановить user_id из
    хеша без знания соли невозможно.
    """
    raw = f"{user_id}:{Config.SESSION_HASH_SALT}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]