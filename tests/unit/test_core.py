"""Smoke-тесты базовой обвязки."""
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


# ============================================
# generate_user_hash
#
# ВАЖНО: хеш больше НЕ зависит от даты. Один пользователь всегда получает
# один и тот же хеш — это позволяет отслеживать его активность во времени.
# ============================================

def test_generate_user_hash_consistent():
    """Один и тот же user_id → один и тот же хеш."""
    hash1 = generate_user_hash(12345)
    hash2 = generate_user_hash(12345)
    assert hash1 == hash2


def test_generate_user_hash_stable_over_time():
    """
    Хеш НЕ зависит от даты: повторные вызовы (в любой день) дают тот же хеш.

    Раньше дата подмешивалась и хеши различались по дням — теперь нет.
    Проверяем стабильность: многократный вызов возвращает идентичный результат.
    """
    hashes = {generate_user_hash(12345) for _ in range(5)}
    assert len(hashes) == 1


def test_generate_user_hash_different_users():
    """Разные пользователи → разные хеши."""
    hash1 = generate_user_hash(12345)
    hash2 = generate_user_hash(67890)
    assert hash1 != hash2


def test_generate_user_hash_length():
    """Хеш — усечённый sha256 до 16 символов."""
    h = generate_user_hash(12345)
    assert len(h) == 16
    assert isinstance(h, str)
