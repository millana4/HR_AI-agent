"""Тесты PII-парсера.

Плейсхолдеры нумерованные: NAME_1, NAME_2, ... (не [NAME]).
"""
from unittest.mock import AsyncMock

import pytest

from app.repositories.nocodb_client import NocoDBClient
from app.services.pii_cache import get_pii_cache
from app.services.pii_parser import (
    NAME_PLACEHOLDER_PREFIX,
    PiiParser,
    make_placeholder,
)


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


def test_make_placeholder():
    """Плейсхолдер формируется по номеру."""
    assert make_placeholder(1) == "NAME_1"
    assert make_placeholder(2) == "NAME_2"
    assert NAME_PLACEHOLDER_PREFIX == "NAME_"


async def test_parse_no_names(parser: PiiParser):
    """Текст без имён остаётся без изменений."""
    result = parser.parse("Где найти бланк отпуска?")
    assert result.masked_text == "Где найти бланк отпуска?"
    assert result.found_names == []


async def test_parse_single_surname(parser: PiiParser):
    """Фамилия в именительном падеже."""
    result = parser.parse("Найди телефон Иванов")
    assert "NAME_1" in result.masked_text
    assert "Иванов" in result.found_names


async def test_parse_surname_genitive(parser: PiiParser):
    """Фамилия в родительном падеже подменяется и нормализуется."""
    result = parser.parse("Найди телефон Иванова")
    assert result.masked_text == "Найди телефон NAME_1"
    assert result.found_names == ["Иванов"]


async def test_parse_first_and_last_name(parser: PiiParser):
    """Имя и фамилия в одном запросе — нумеруются по порядку."""
    result = parser.parse("Найди контакт Петрова Сидора")
    assert result.masked_text == "Найди контакт NAME_1 NAME_2"
    assert "Пётр" in result.found_names or "Петров" in result.found_names
    assert "Сидор" in result.found_names or "Сидоров" in result.found_names


async def test_parse_case_insensitive(parser: PiiParser):
    """Регистр не влияет на матчинг."""
    result = parser.parse("найди телефон ИВАНОВА")
    assert "NAME_1" in result.masked_text
    assert "Иванов" in result.found_names


async def test_parse_preserves_punctuation(parser: PiiParser):
    """Пунктуация сохраняется в маскированном тексте."""
    result = parser.parse("Привет! Найди, пожалуйста, Иванова.")
    assert result.masked_text == "Привет! Найди, пожалуйста, NAME_1."
    assert result.found_names == ["Иванов"]


async def test_parse_patronymic_not_matched(parser: PiiParser):
    """Отчество (Иванович) НЕ должно матчиться — оно не в словаре."""
    result = parser.parse("Найди Ивановича")
    assert result.masked_text == "Найди Ивановича"
    assert result.found_names == []


async def test_parse_unknown_name_not_masked(parser: PiiParser):
    """Имя, которого нет в словаре сотрудников, не маскируется."""
    result = parser.parse("Найди телефон Распутина")
    assert "NAME_" not in result.masked_text
    assert result.found_names == []


async def test_parse_multiple_occurrences(parser: PiiParser):
    """Несколько имён в одном запросе — каждое со своим номером."""
    result = parser.parse("Скажи Иванову про Петрова")
    assert result.masked_text == "Скажи NAME_1 про NAME_2"
    assert result.found_names == ["Иванов", "Пётр"] or result.found_names == ["Иванов", "Петров"]


async def test_parse_numbering_in_order(parser: PiiParser):
    """Номера плейсхолдеров идут по порядку появления, начиная с 1."""
    result = parser.parse("Иванов Петров Сидоров")
    # три разных имени → NAME_1 NAME_2 NAME_3
    assert "NAME_1" in result.masked_text
    assert "NAME_2" in result.masked_text
    assert "NAME_3" in result.masked_text
    assert len(result.found_names) == 3


async def test_parse_short_name(parser: PiiParser, monkeypatch):
    """Короткое имя тоже распознаётся, если есть в словаре."""
    async def fake_fetch(client, correlation_id="-"):
        return ["Янов Ян Янович"]

    monkeypatch.setattr("app.services.pii_cache.fetch_pivot", fake_fetch)
    cache = get_pii_cache()
    cache._forms = {}
    cache._built_for_date = None

    p = PiiParser()
    await p.ensure_ready(NocoDBClient())
    result = p.parse("Скажи Яну привет")
    assert "NAME_1" in result.masked_text
    assert "Ян" in result.found_names