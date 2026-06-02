"""
Утилиты безопасности: маскирование PII, хеширование пользовательских сессий.
"""
import hashlib
import re
from datetime import date

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


def generate_user_hash(user_id: int, target_date: date | None = None) -> str:
    """
    Генерирует хеш сессии для записи в аналитику.

    Хеш зависит от (user_id, date, salt). Один пользователь за один день — один хеш.
    На следующий день у того же пользователя будет другой хеш.

    Восстановить user_id из хеша без знания соли невозможно.
    """
    if target_date is None:
        target_date = date.today()

    raw = f"{user_id}:{target_date.isoformat()}:{Config.SESSION_HASH_SALT}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]