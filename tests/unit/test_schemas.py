"""Тесты pydantic-моделей API."""
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AskRequest,
    TextResponse,
    ToolCall,
    ToolCallResponse,
    ErrorResponse,
)


# ============================================
# AskRequest
# ============================================

def test_ask_request_minimal():
    """Минимальный валидный запрос — без correlation_id."""
    req = AskRequest(user_id=123, request="Привет")
    assert req.user_id == 123
    assert req.request == "Привет"
    assert req.correlation_id is None


def test_ask_request_full():
    """Полный запрос со всеми полями."""
    req = AskRequest(user_id=123, request="Привет", correlation_id="abc123")
    assert req.correlation_id == "abc123"


def test_ask_request_empty_text_rejected():
    """Пустой текст запроса отклоняется."""
    with pytest.raises(ValidationError):
        AskRequest(user_id=123, request="")


def test_ask_request_too_long_text_rejected():
    """Слишком длинный текст отклоняется."""
    with pytest.raises(ValidationError):
        AskRequest(user_id=123, request="a" * 2001)


def test_ask_request_missing_user_id_rejected():
    """user_id обязателен."""
    with pytest.raises(ValidationError):
        AskRequest(request="Привет")


# ============================================
# TextResponse
# ============================================

def test_text_response_valid():
    """Валидный текстовый ответ."""
    resp = TextResponse(
        answer="Ответ",
        tool_used="search_faq",
        correlation_id="abc123",
    )
    assert resp.response_type == "text"
    assert resp.tool_used == "search_faq"


def test_text_response_wrong_tool_rejected():
    """В tool_used нельзя положить bot-command."""
    with pytest.raises(ValidationError):
        TextResponse(
            answer="Ответ",
            tool_used="search_contacts",  # это bot-command, не агентский
            correlation_id="abc123",
        )


# ============================================
# ToolCall
# ============================================

def test_tool_call_with_args():
    """Tool call с аргументами."""
    tc = ToolCall(
        name="search_contacts",
        args={"first_name": "Иван", "last_name": "Иванов"},
    )
    assert tc.name == "search_contacts"
    assert tc.args["last_name"] == "Иванов"


def test_tool_call_empty_args():
    """Tool call без аргументов — например suggest_hr_form."""
    tc = ToolCall(name="suggest_hr_form")
    assert tc.args == {}


def test_tool_call_unknown_name_rejected():
    """Неизвестное имя tool отклоняется."""
    with pytest.raises(ValidationError):
        ToolCall(name="search_unknown", args={})


# ============================================
# ToolCallResponse
# ============================================

def test_tool_call_response_single():
    """Ответ с одним tool call."""
    resp = ToolCallResponse(
        tool_calls=[
            ToolCall(name="search_contacts", args={"last_name": "Иванов"})
        ],
        correlation_id="abc123",
    )
    assert resp.response_type == "tool_call"
    assert len(resp.tool_calls) == 1


def test_tool_call_response_multiple():
    """Ответ с несколькими tool calls — структура заложена под это."""
    resp = ToolCallResponse(
        tool_calls=[
            ToolCall(name="search_contacts", args={"last_name": "Иванов"}),
            ToolCall(name="search_ats_mavis", args={"department": "Бухгалтерия"}),
        ],
        correlation_id="abc123",
    )
    assert len(resp.tool_calls) == 2


def test_tool_call_response_empty_list_rejected():
    """Пустой список tool_calls отклоняется."""
    with pytest.raises(ValidationError):
        ToolCallResponse(tool_calls=[], correlation_id="abc123")


# ============================================
# ErrorResponse
# ============================================

def test_error_response_minimal():
    """Минимальная ошибка."""
    err = ErrorResponse(error="LLM timeout", correlation_id="abc123")
    assert err.error == "LLM timeout"
    assert err.detail is None


def test_error_response_with_detail():
    """Ошибка с подробностями."""
    err = ErrorResponse(
        error="LLM timeout",
        detail="External LLM did not respond within 60 seconds",
        correlation_id="abc123",
    )
    assert err.detail is not None