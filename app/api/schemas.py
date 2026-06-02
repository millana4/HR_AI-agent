"""
Pydantic-модели для контракта между ботом и агентом.

Соответствует docs/openapi.yaml.
"""
from typing import Literal

from pydantic import BaseModel, Field


# ============================================
# Запрос
# ============================================

class AskRequest(BaseModel):
    """Запрос от бота к агенту."""

    user_id: int = Field(
        description="Telegram ID пользователя",
        examples=[123456789],
    )
    request: str = Field(
        min_length=1,
        max_length=2000,
        description="Текст вопроса пользователя",
        examples=["Где найти бланк заявления на отпуск?"],
    )
    correlation_id: str | None = Field(
        default=None,
        description="Идентификатор трассировки. Если не передан — агент сгенерирует.",
    )


# ============================================
# Tool call
# ============================================

ToolName = Literal[
    "search_contacts",
    "search_ats_mavis",
    "search_ats_votonia",
    "search_shop",
    "search_drugstore",
    "suggest_hr_form",
]


class ToolCall(BaseModel):
    """Одна инструкция боту: имя команды + аргументы."""

    name: ToolName = Field(description="Имя команды для бота")
    args: dict = Field(
        default_factory=dict,
        description="Аргументы команды. Структура зависит от name.",
    )


# ============================================
# Ответ
# ============================================

AgentToolName = Literal[
    "search_faq",
    "search_documents",
    "search_knowledge_base",
    "answer_general",
]


class TextResponse(BaseModel):
    """Текстовый ответ — готовый текст для пользователя."""

    response_type: Literal["text"] = "text"
    answer: str = Field(description="Текст ответа для пользователя")
    tool_used: AgentToolName = Field(
        description="Какой внутренний tool агента сформировал ответ"
    )
    correlation_id: str


class ToolCallResponse(BaseModel):
    """Tool call — инструкция боту выполнить действия."""

    response_type: Literal["tool_call"] = "tool_call"
    tool_calls: list[ToolCall] = Field(
        min_length=1,
        description="Список действий для выполнения ботом",
    )
    correlation_id: str

# Type alias для ответа эндпоинта /api/ask
AskResponse = TextResponse | ToolCallResponse


# ============================================
# Ошибка
# ============================================

class ErrorResponse(BaseModel):
    """Стандартный формат ошибки."""

    error: str = Field(description="Короткое описание ошибки")
    detail: str | None = Field(
        default=None,
        description="Развёрнутое описание ошибки",
    )
    correlation_id: str


# ============================================
# Health
# ============================================

class HealthResponse(BaseModel):
    """Ответ эндпоинта /api/health."""

    status: Literal["ok"] = "ok"
    version: str = "1.0.0"