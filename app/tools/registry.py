"""
Реестр инструментов агента.

Здесь описаны все tools: их JSON-схемы для LLM и тип выполнения.

Tools бывают двух типов:
- agent_internal: агент выполняет инструмент сам (поиск в Qdrant),
  затем формирует финальный текстовый ответ.
- bot_command: агент НЕ выполняет, а возвращает команду боту в формате
  ToolCallResponse. Бот сам сходит за данными и сформирует ответ пользователю.

answer_general — особый случай: это НЕ tool в реестре, а ветка в agent_loop
(LLM ответила текстом без вызова tool).
"""
from dataclasses import dataclass
from typing import Literal

from app.api.schemas import AgentToolName
from app.llm.base import ToolSpec


ToolKind = Literal["agent_internal", "bot_command"]


@dataclass(frozen=True)
class RegisteredTool:
    """Описание одного tool: имя, тип, спецификация для LLM."""

    name: AgentToolName | str  # str для bot-command tools (их имена не в AgentToolName)
    kind: ToolKind
    spec: ToolSpec


# ---------- Спецификации tools для LLM ----------

_SEARCH_INTERNAL_SPEC = ToolSpec(
    name="search_internal",
    description=(
        "Поиск во ВСЕХ внутренних источниках компании сразу: база вопросов-ответов "
        "(FAQ), официальные документы (регламенты, инструкции, велкомбук, бланки "
        "заявлений) и база знаний по бизнес-процессам (ЭДО СБИС и Корус, Честный "
        "знак, маркетплейсы, карточки товаров в 1С, подключение поставщиков). "
        "Используй для ЛЮБОГО вопроса о работе в компании: график, отпуск, "
        "больничный, корпоратив, мероприятия, процедуры, бланки, регламенты, "
        "IT-процессы. Ответ может быть в любом из источников — не выбирай источник "
        "сам, этот инструмент ищет везде."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Поисковый запрос на русском языке",
            },
        },
        "required": ["query"],
    },
)

_SEARCH_CONTACTS_SPEC = ToolSpec(
    name="search_contacts",
    description=(
        "Найти контакты сотрудников компании по ФИО. "
        "Использовать, когда пользователь хочет найти конкретного человека."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Имя и/или фамилия",
            },
        },
        "required": ["query"],
    },
)

_SEARCH_ATS_MAVIS_SPEC = ToolSpec(
    name="search_ats_mavis",
    description=(
        "Найти внутренние телефоны отделов компании Мавис."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Название отдела",
            },
        },
        "required": ["query"],
    },
)

_SEARCH_ATS_VOTONIA_SPEC = ToolSpec(
    name="search_ats_votonia",
    description=(
        "Найти внутренние телефоны отделов компании Вотоня."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Название отдела",
            },
        },
        "required": ["query"],
    },
)

_SEARCH_SHOP_SPEC = ToolSpec(
    name="search_shop",
    description=(
        "Найти контакты магазина Вотоня по адресу"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Улица, где находится магазин",
            },
        },
        "required": ["query"],
    },
)

_SEARCH_DRUGSTORE_SPEC = ToolSpec(
    name="search_drugstore",
    description=(
        "Найти контакты аптеки по адресу."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Улица, где находится аптека",
            },
        },
        "required": ["query"],
    },
)

_SUGGEST_HR_FORM_SPEC = ToolSpec(
    name="suggest_hr_form",
    description=(
        "Предложить пользователю заполнить форму обращения в HR-отдел. "
        "Использовать когда у пользователя сложный вопрос, на который не может "
        "ответить ни одно из других средств, и этот вопрос касается внутренней "
        "работы компании и нужна помощь живого HR."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)


# ---------- Реестр ----------

TOOLS: dict[str, RegisteredTool] = {
    # Agent-internal: агент сам ищет во всех внутренних источниках,
    # потом формирует ответ из контекста.
    "search_internal": RegisteredTool("search_internal", "agent_internal", _SEARCH_INTERNAL_SPEC),

    # Bot-command: возвращаем боту, он сам выполнит.
    "search_contacts": RegisteredTool("search_contacts", "bot_command", _SEARCH_CONTACTS_SPEC),
    "search_ats_mavis": RegisteredTool("search_ats_mavis", "bot_command", _SEARCH_ATS_MAVIS_SPEC),
    "search_ats_votonia": RegisteredTool("search_ats_votonia", "bot_command", _SEARCH_ATS_VOTONIA_SPEC),
    "search_shop": RegisteredTool("search_shop", "bot_command", _SEARCH_SHOP_SPEC),
    "search_drugstore": RegisteredTool("search_drugstore", "bot_command", _SEARCH_DRUGSTORE_SPEC),
    "suggest_hr_form": RegisteredTool("suggest_hr_form", "bot_command", _SUGGEST_HR_FORM_SPEC),
}


def get_all_tool_specs() -> list[ToolSpec]:
    """Все спецификации tools для передачи LLM."""
    return [tool.spec for tool in TOOLS.values()]


def get_tool_kind(tool_name: str) -> ToolKind | None:
    """Тип tool по имени или None если такого tool нет."""
    tool = TOOLS.get(tool_name)
    return tool.kind if tool else None


def is_agent_internal(tool_name: str) -> bool:
    """True если tool агент выполняет сам."""
    return get_tool_kind(tool_name) == "agent_internal"


def is_bot_command(tool_name: str) -> bool:
    """True если tool нужно вернуть боту для выполнения."""
    return get_tool_kind(tool_name) == "bot_command"