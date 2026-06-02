"""Smoke-тесты базовой обвязки."""
from datetime import date

from app.core.security import generate_user_hash, mask_pii


def test_mask_pii_phone():
    text = "Звоните +7 (911) 123-45-67"
    masked = mask_pii(text)
    assert "[PHONE]" in masked
    assert "911" not in masked


def test_mask_pii_email():
    text = "Пишите на ivanov@mavis.ru"
    masked = mask_pii(text)
    assert "[EMAIL]" in masked
    assert "ivanov" not in masked


def test_mask_pii_empty():
    assert mask_pii("") == ""


def test_generate_user_hash_consistent():
    """Один и тот же user+date → один хеш."""
    target = date(2026, 6, 2)
    hash1 = generate_user_hash(12345, target)
    hash2 = generate_user_hash(12345, target)
    assert hash1 == hash2


def test_generate_user_hash_different_days():
    """Разные даты → разные хеши."""
    hash1 = generate_user_hash(12345, date(2026, 6, 2))
    hash2 = generate_user_hash(12345, date(2026, 6, 3))
    assert hash1 != hash2


def test_generate_user_hash_different_users():
    """Разные пользователи → разные хеши."""
    target = date(2026, 6, 2)
    hash1 = generate_user_hash(12345, target)
    hash2 = generate_user_hash(67890, target)
    assert hash1 != hash2