"""
Юнит-тесты ключевых хелперов agent_common: разбор и подстановка скрытых
данных (защита ПД). Покрываем только критичную логику, по минимуму.
"""
from app.services.agent_common import (
    parse_hidden_data,
    substitute_hidden_data,
)


# ============================================
# parse_hidden_data
# ============================================

def test_parse_hidden_data_single():
    """Одна пара ИМЯ=значение."""
    result = parse_hidden_data(["АХО_КОНТАКТ=Bosova.V@mavis.ru"])
    assert result == {"АХО_КОНТАКТ": "Bosova.V@mavis.ru"}


def test_parse_hidden_data_multiline():
    """Несколько пар в одной строке через перевод строки."""
    result = parse_hidden_data(["A=1\nB=2"])
    assert result == {"A": "1", "B": "2"}


def test_parse_hidden_data_multiple_entries():
    """Несколько строк из разных чанков."""
    result = parse_hidden_data(["A=1", "B=2"])
    assert result == {"A": "1", "B": "2"}


def test_parse_hidden_data_skips_garbage():
    """Пустые и строки без '=' пропускаются."""
    result = parse_hidden_data(["", "мусор без равно", "A=1"])
    assert result == {"A": "1"}


# ============================================
# substitute_hidden_data
# ============================================

def test_substitute_replaces_placeholder():
    """#ИМЯ заменяется на значение."""
    text = "Пишите на #АХО_КОНТАКТ за справкой."
    result = substitute_hidden_data(text, {"АХО_КОНТАКТ": "Bosova.V@mavis.ru"})
    assert result == "Пишите на Bosova.V@mavis.ru за справкой."


def test_substitute_orphan_placeholder_loses_hash():
    """Плейсхолдер без значения в map → решётка убирается, остаётся слово."""
    text = "Контакт: #НЕИЗВЕСТНО."
    result = substitute_hidden_data(text, {})
    assert "#" not in result
    assert "НЕИЗВЕСТНО" in result


def test_substitute_leaves_abbreviations_untouched():
    """Аббревиатуры без решётки (ГПХ, НДФЛ) не трогаются."""
    text = "Договор ГПХ и вычет НДФЛ оформляются отдельно."
    result = substitute_hidden_data(text, {"КОНТАКТ": "x@mavis.ru"})
    assert result == text


def test_substitute_empty_text():
    """Пустой текст не ломается."""
    assert substitute_hidden_data("", {"A": "1"}) == ""