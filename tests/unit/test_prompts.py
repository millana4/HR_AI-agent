"""Тесты системных промптов."""
from app.llm.prompts_gigachat import (
    CONTEXT_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    make_context_prompt,
)


def test_system_prompt_mentions_all_tools():
    """В промпте упомянуты все 9 tools (для триггера правильных вызовов)."""
    tools = [
        "search_faq",
        "search_documents",
        "search_wiki",
        "search_contacts",
        "search_ats_mavis",
        "search_ats_votonia",
        "search_shop",
        "search_drugstore",
        "suggest_hr_form",
    ]
    for tool in tools:
        assert tool in SYSTEM_PROMPT, f"Tool {tool} not mentioned in system prompt"


def test_system_prompt_mentions_pii_placeholder():
    """PII-плейсхолдер [NAME] должен быть упомянут — иначе LLM может не понять."""
    assert "[NAME]" in SYSTEM_PROMPT


def test_system_prompt_forbids_markdown():
    """Промпт должен запрещать markdown — для Telegram."""
    assert "markdown" in SYSTEM_PROMPT.lower()


def test_system_prompt_in_russian():
    """Промпт на русском — критично для GigaChat."""
    # Считаем кириллические буквы — их должно быть много
    cyrillic_count = sum(1 for ch in SYSTEM_PROMPT if "а" <= ch.lower() <= "я")
    assert cyrillic_count > 500


def test_make_context_prompt_includes_tool_and_context():
    """Контекстный промпт включает имя tool и сам контекст."""
    prompt = make_context_prompt(
        tool_name="search_faq",
        context="Вопрос: X. Ответ: Y.",
    )
    assert "search_faq" in prompt
    assert "Вопрос: X. Ответ: Y." in prompt


def test_make_context_prompt_with_empty_context():
    """Пустой контекст не ломает форматирование."""
    prompt = make_context_prompt(tool_name="search_faq", context="")
    assert "search_faq" in prompt
    # Шаблон не должен падать на пустой строке
    assert isinstance(prompt, str)


def test_context_prompt_template_has_required_placeholders():
    """Шаблон содержит обе подстановки."""
    assert "{tool_name}" in CONTEXT_PROMPT_TEMPLATE
    assert "{context}" in CONTEXT_PROMPT_TEMPLATE