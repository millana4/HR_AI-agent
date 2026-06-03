"""Тесты PII-парсера."""
from unittest.mock import AsyncMock

import pytest

from app.repositories.nocodb_client import NocoDBClient
from app.services.pii_cache import get_pii_cache
from app.services.pii_parser import NAME_PLACEHOLDER, PiiParser


@pytest.fixture
async def parser(monkeypatch):
    """Парсер с замоканной NocoDB-выгрузкой."""
    async def fake_fetch(client, correlation_id="-"):
        return ["Иванов Иван Иванович", "Петров Пётр Петрович", "Сидоров Сидор"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)

    # Сбрасываем глобальный кеш, чтобы тесты не влияли друг на друга
    cache = get_pii_cache()
    cache._forms = {}
    cache._built_for_date = None

    p = PiiParser()
    await p.ensure_ready(NocoDBClient())
    return p


async def test_parse_no_names(parser: PiiParser):
    """Текст без имён остаётся без изменений."""
    result = parser.parse("Где найти бланк отпуска?")
    assert result.masked_text == "Где найти бланк отпуска?"
    assert result.found_names == []


async def test_parse_single_surname(parser: PiiParser):
    """Фамилия в именительном падеже."""
    result = parser.parse("Найди телефон Иванов")
    assert NAME_PLACEHOLDER in result.masked_text
    assert "Иванов" in result.found_names


async def test_parse_surname_genitive(parser: PiiParser):
    """Фамилия в родительном падеже подменяется и нормализуется."""
    result = parser.parse("Найди телефон Иванова")
    assert result.masked_text == "Найди телефон [NAME]"
    assert result.found_names == ["Иванов"]


async def test_parse_first_and_last_name(parser: PiiParser):
    """Имя и фамилия в одном запросе."""
    result = parser.parse("Найди контакт Иванова Ивана")
    assert result.masked_text == "Найди контакт [NAME] [NAME]"
    assert "Иванов" in result.found_names
    assert "Иван" in result.found_names


async def test_parse_case_insensitive(parser: PiiParser):
    """Регистр не влияет на матчинг."""
    result = parser.parse("найди телефон ИВАНОВА")
    assert "[NAME]" in result.masked_text
    assert "Иванов" in result.found_names


async def test_parse_preserves_punctuation(parser: PiiParser):
    """Пунктуация сохраняется в маскированном тексте."""
    result = parser.parse("Привет! Найди, пожалуйста, Иванова.")
    assert result.masked_text == "Привет! Найди, пожалуйста, [NAME]."
    assert result.found_names == ["Иванов"]


async def test_parse_patronymic_not_matched(parser: PiiParser):
    """Отчество (Иванович) НЕ должно матчиться — оно не в словаре."""
    result = parser.parse("Найди Ивановича")
    # Слово "Ивановича" — отчество в род. падеже, в словаре его нет
    assert result.masked_text == "Найди Ивановича"
    assert result.found_names == []


async def test_parse_unknown_name_not_masked(parser: PiiParser):
    """Имя, которого нет в словаре сотрудников, не маскируется."""
    result = parser.parse("Найди телефон Распутина")
    assert "[NAME]" not in result.masked_text
    assert result.found_names == []


async def test_parse_multiple_occurrences(parser: PiiParser):
    """Несколько имён в одном запросе."""
    result = parser.parse("Скажи Иванову про Петрова")
    assert result.masked_text == "Скажи [NAME] про [NAME]"
    assert result.found_names == ["Иванов", "Пётр"] or result.found_names == ["Иванов", "Петров"]


async def test_parse_short_name(parser: PiiParser, monkeypatch):
    """Короткое имя тоже распознаётся, если есть в словаре."""
    # Подмешиваем сотрудника с коротким именем
    async def fake_fetch(client, correlation_id="-"):
        return ["Янов Ян Янович"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)
    cache = get_pii_cache()
    cache._forms = {}
    cache._built_for_date = None

    p = PiiParser()
    await p.ensure_ready(NocoDBClient())
    result = p.parse("Скажи Яну привет")
    assert "[NAME]" in result.masked_text
    assert "Ян" in result.found_names