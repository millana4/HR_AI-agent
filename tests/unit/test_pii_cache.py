"""Тесты кеша словоформ ФИО."""
from datetime import date, timedelta
from unittest.mock import AsyncMock

import pytest

from app.repositories.nocodb_client import NocoDBClient
from app.services.pii_cache import PiiCache


@pytest.fixture
def mock_client():
    """Клиент NocoDB с замоканным fetch_pivot."""
    return NocoDBClient()


async def test_cache_builds_on_first_call(mock_client, monkeypatch):
    """Первый вызов ensure_fresh — кеш строится."""
    async def fake_fetch(client, correlation_id="-"):
        return ["Иванов Иван Иванович"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)

    cache = PiiCache()
    await cache.ensure_fresh(mock_client)
    forms = cache.get_forms()

    # Должны быть словоформы для "Иванов" и "Иван"
    assert "иванов" in forms
    assert "иванова" in forms  # склонение
    assert "иван" in forms
    assert "ивана" in forms

    # Леммы в именительном с заглавной
    assert forms["иванов"] == "Иванов"
    assert forms["иванова"] == "Иванов"


async def test_cache_skips_patronymic(mock_client, monkeypatch):
    """Отчество не должно попасть в словарь."""
    async def fake_fetch(client, correlation_id="-"):
        return ["Иванов Иван Сидорович"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)

    cache = PiiCache()
    await cache.ensure_fresh(mock_client)
    forms = cache.get_forms()

    # "Иванов" и "Иван" — есть. Любое словоформы "Сидорович" — нет
    assert "иванов" in forms
    assert "иван" in forms
    # Отчество "Сидорович" в любых формах не должно быть
    assert "сидорович" not in forms
    assert "сидоровича" not in forms


async def test_cache_handles_two_word_fio(mock_client, monkeypatch):
    """Если в ФИО только два слова — оба берутся в словарь."""
    async def fake_fetch(client, correlation_id="-"):
        return ["Петров Пётр"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)

    cache = PiiCache()
    await cache.ensure_fresh(mock_client)
    forms = cache.get_forms()

    assert "петров" in forms
    assert "пётр" in forms or "петр" in forms


async def test_cache_not_rebuilt_same_day(mock_client, monkeypatch):
    """Второй вызов в тот же день — не перестраивает."""
    call_count = 0

    async def fake_fetch(client, correlation_id="-"):
        nonlocal call_count
        call_count += 1
        return ["Иванов Иван Иванович"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)

    cache = PiiCache()
    await cache.ensure_fresh(mock_client)
    await cache.ensure_fresh(mock_client)
    await cache.ensure_fresh(mock_client)

    assert call_count == 1


async def test_cache_rebuilt_on_new_day(mock_client, monkeypatch):
    """Если кеш построен вчера — сегодня перестраивается."""
    call_count = 0

    async def fake_fetch(client, correlation_id="-"):
        nonlocal call_count
        call_count += 1
        return ["Иванов Иван Иванович"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)

    cache = PiiCache()
    await cache.ensure_fresh(mock_client)
    assert call_count == 1

    # Подделываем дату построения как "вчера"
    cache._built_for_date = date.today() - timedelta(days=1)
    await cache.ensure_fresh(mock_client)
    assert call_count == 2


async def test_cache_case_insensitive_keys(mock_client, monkeypatch):
    """Все ключи в словаре — в нижнем регистре."""
    async def fake_fetch(client, correlation_id="-"):
        return ["ИВАНОВ Иван Иванович", "петров Пётр"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)

    cache = PiiCache()
    await cache.ensure_fresh(mock_client)
    forms = cache.get_forms()

    # Все ключи должны быть lowercase, независимо от исходного регистра
    for key in forms:
        assert key == key.lower(), f"Ключ {key} не в lowercase"