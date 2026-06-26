"""Тесты реестра tools.

После миграции на Yandex-ветку в реестре 9 tools и 4 kind:
  - agent_internal: search_internal
  - bot_command:    search_contacts, search_ats_*, search_shop,
                    search_drugstore, suggest_hr_form
  - agent_general:  answer_general
  - agent_image:    generate_image
"""
from app.tools.registry import (
    TOOLS,
    get_all_tool_names,
    get_all_tool_specs,
    get_tool_kind,
    is_agent_general,
    is_agent_image,
    is_agent_internal,
    is_bot_command,
)


_ALL_KINDS = ("agent_internal", "bot_command", "agent_general", "agent_image")


def test_all_expected_tools_registered():
    """В реестре должны быть все 9 ожидаемых tools."""
    expected = {
        "search_internal",
        "search_contacts", "search_ats_mavis", "search_ats_votonia",
        "search_shop", "search_drugstore", "suggest_hr_form",
        "answer_general", "generate_image",
    }
    assert set(TOOLS.keys()) == expected


def test_agent_internal_tools():
    """search_internal — единственный внутренний tool."""
    assert is_agent_internal("search_internal")
    assert not is_bot_command("search_internal")
    assert not is_agent_general("search_internal")
    assert not is_agent_image("search_internal")


def test_bot_command_tools():
    """Контакты/телефоны/магазины/аптеки/форма — bot-command."""
    for name in (
        "search_contacts", "search_ats_mavis", "search_ats_votonia",
        "search_shop", "search_drugstore", "suggest_hr_form",
    ):
        assert is_bot_command(name), f"{name} должен быть bot_command"
        assert not is_agent_internal(name)


def test_agent_general_tool():
    """answer_general — отдельный kind agent_general."""
    assert is_agent_general("answer_general")
    assert not is_bot_command("answer_general")
    assert not is_agent_internal("answer_general")
    assert not is_agent_image("answer_general")


def test_agent_image_tool():
    """generate_image — отдельный kind agent_image."""
    assert is_agent_image("generate_image")
    assert not is_agent_general("generate_image")
    assert not is_bot_command("generate_image")
    assert not is_agent_internal("generate_image")


def test_every_tool_has_exactly_one_kind():
    """Каждый tool принадлежит ровно одному из четырёх kind."""
    for name in TOOLS:
        kind = get_tool_kind(name)
        assert kind in _ALL_KINDS, f"{name}: неизвестный kind {kind}"
        # Ровно один из предикатов истинен.
        flags = [
            is_agent_internal(name),
            is_bot_command(name),
            is_agent_general(name),
            is_agent_image(name),
        ]
        assert sum(flags) == 1, f"{name}: должен быть ровно один kind, а не {flags}"


def test_unknown_tool_returns_none():
    assert get_tool_kind("nonexistent_tool") is None
    assert not is_agent_internal("nonexistent_tool")
    assert not is_bot_command("nonexistent_tool")
    assert not is_agent_general("nonexistent_tool")
    assert not is_agent_image("nonexistent_tool")


def test_get_all_tool_specs_returns_all():
    specs = get_all_tool_specs()
    assert len(specs) == len(TOOLS)
    names = {spec.name for spec in specs}
    assert names == set(TOOLS.keys())


def test_get_all_tool_names():
    """get_all_tool_names возвращает множество всех имён."""
    names = get_all_tool_names()
    assert names == set(TOOLS.keys())
    assert "answer_general" in names
    assert "generate_image" in names


def test_each_spec_has_required_fields():
    """Каждая спецификация имеет name, description, parameters."""
    for spec in get_all_tool_specs():
        assert spec.name
        assert spec.description
        assert isinstance(spec.parameters, dict)
        assert "type" in spec.parameters


def test_specs_have_russian_descriptions():
    """Описания на русском — критично для качества вызова."""
    for spec in get_all_tool_specs():
        assert any("а" <= ch.lower() <= "я" for ch in spec.description), (
            f"Tool {spec.name} description should be in Russian"
        )