"""
Абстракция LLM-клиента.

Определяет контракт работы с любой внешней LLM (GigaChat, MiniMax, OpenAI, ...).
Конкретные реализации лежат в app/llm/<provider>.py.

Двухпроходная схема использования (в agent_loop, шаг 12):
    Pass 1: chat(messages, tools=ALL_TOOLS) — классификация, LLM выбирает tool
    Pass 2: chat(messages_with_context) — финал, LLM формирует ответ для пользователя
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """Одно сообщение в истории разговора с LLM."""

    role: Role
    content: str


@dataclass
class ToolSpec:
    """
    Описание tool, которое передаётся LLM, чтобы она знала о существовании tool.

    Соответствует JSON Schema для function calling.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema объект


@dataclass
class ToolCall:
    """LLM просит вызвать tool с такими-то аргументами."""

    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    """
    Ответ от LLM в нашем внутреннем нормализованном формате.

    Один из двух типов:
    - text: LLM ответила текстом, без вызова tool
    - tool_calls: LLM хочет вызвать один или несколько tools
    """

    type: Literal["text", "tool_calls"]
    content: str = ""  # заполнено для type="text"
    tool_calls: list[ToolCall] = field(default_factory=list)  # для type="tool_calls"


class BaseLLMClient(ABC):
    """
    Базовый класс LLM-клиента.

    Конкретные реализации должны:
    - Реализовать метод chat()
    - Переводить из внутреннего формата (Message, ToolSpec) в формат своего API
    - Парсить ответ API в LLMResponse
    - Кидать LLMTimeoutError при таймауте, LLMError при прочих ошибках
    """

    @abstractmethod
    async def chat(
            self,
            messages: list[Message],
            tools: list[ToolSpec] | None = None,
            correlation_id: str = "-",
            model: str | None = None,
    ) -> LLMResponse:
        """
        Отправить запрос в LLM.

        Args:
            messages: история разговора (system + user + assistant)
            tools: список доступных tools для function calling. Опционально.
            correlation_id: для трассировки в логах.
            model: какую модель использовать. Если None — реализация берёт свою по умолчанию.

        Returns:
            LLMResponse с типом text или tool_calls.

        Raises:
            LLMTimeoutError: LLM не ответила за Config.LLM_TIMEOUT секунд.
            LLMError: прочие ошибки (5xx от API, невалидный ответ и т.д.).
        """