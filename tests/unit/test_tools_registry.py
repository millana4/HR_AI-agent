
"""Тесты реестра tools."""
from app.tools.registry import (
    TOOLS,
    get_all_tool_specs,
    get_tool_kind,
    is_agent_internal,
    is_bot_command,
)


def test_all_expected_tools_registered():
    """В реестре должны быть все 7 ожидаемых tools."""
    expected = {
        "search_internal",
        "search_contacts", "search_ats_mavis", "search_ats_votonia",
        "search_shop", "search_drugstore", "suggest_hr_form",
    }
    assert set(TOOLS.keys()) == expected


def test_agent_internal_tools():
    """search_internal — единственный внутренний tool."""
    assert is_agent_internal("search_internal")


def test_bot_command_tools():
    """Все остальные — bot-command."""
    assert is_bot_command("search_contacts")
    assert is_bot_command("search_ats_mavis")
    assert is_bot_command("search_ats_votonia")
    assert is_bot_command("search_shop")
    assert is_bot_command("search_drugstore")
    assert is_bot_command("suggest_hr_form")


def test_internal_and_bot_command_mutually_exclusive():
    """Tool не может быть одновременно agent_internal и bot_command."""
    for name in TOOLS:
        kind = get_tool_kind(name)
        assert kind in ("agent_internal", "bot_command")
        if kind == "agent_internal":
            assert is_agent_internal(name)
            assert not is_bot_command(name)
        else:
            assert not is_agent_internal(name)
            assert is_bot_command(name)


def test_unknown_tool_returns_none():
    assert get_tool_kind("nonexistent_tool") is None
    assert not is_agent_internal("nonexistent_tool")
    assert not is_bot_command("nonexistent_tool")


def test_get_all_tool_specs_returns_all():
    specs = get_all_tool_specs()
    assert len(specs) == len(TOOLS)
    names = {spec.name for spec in specs}
    assert names == set(TOOLS.keys())


def test_each_spec_has_required_fields():
    """Каждая спецификация имеет name, description, parameters."""
    for spec in get_all_tool_specs():
        assert spec.name
        assert spec.description
        assert isinstance(spec.parameters, dict)
        assert "type" in spec.parameters


def test_specs_have_russian_descriptions():
    """Описания на русском — критично для качества вызова в GigaChat."""
    for spec in get_all_tool_specs():
        assert any("а" <= ch.lower() <= "я" for ch in spec.description), (
            f"Tool {spec.name} description should be in Russian"
        )