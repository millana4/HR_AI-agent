"""Тесты внутренних структур LLM-клиента."""
from app.llm.base import LLMResponse, Message, ToolCall, ToolSpec


def test_message_creation():
    msg = Message(role="user", content="Привет")
    assert msg.role == "user"
    assert msg.content == "Привет"


def test_tool_spec_creation():
    spec = ToolSpec(
        name="search_faq",
        description="Поиск в FAQ",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    assert spec.name == "search_faq"
    assert "query" in spec.parameters["properties"]


def test_tool_call_creation():
    call = ToolCall(name="search_faq", args={"query": "отпуск"})
    assert call.name == "search_faq"
    assert call.args["query"] == "отпуск"


def test_llm_response_text():
    resp = LLMResponse(type="text", content="Ответ от LLM")
    assert resp.type == "text"
    assert resp.content == "Ответ от LLM"
    assert resp.tool_calls == []  # default factory


def test_llm_response_tool_calls():
    resp = LLMResponse(
        type="tool_calls",
        tool_calls=[
            ToolCall(name="search_faq", args={"query": "отпуск"}),
            ToolCall(name="search_documents", args={"query": "бланк"}),
        ],
    )
    assert resp.type == "tool_calls"
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].name == "search_faq"
    assert resp.content == ""  # default