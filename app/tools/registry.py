"""
Реестр инструментов агента.

Здесь описаны все tools: их JSON-схемы для LLM и тип выполнения.

Tools бывают двух типов:
- agent_internal: агент выполняет инструмент сам (поиск в Qdrant и т.д.),
  затем формирует финальный текстовый ответ.
- bot_command: агент НЕ выполняет, а возвращает команду боту в формате
  ToolCallResponse. Бот сам сходит за данными и сформирует ответ пользователю.

answer_general — особый случай: это agent_internal "tool", означающий
"ничего не искать, отвечать на знаниях LLM". Реализован прямо в agent_loop
как ветка без tool_call.
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

_SEARCH_FAQ_SPEC = ToolSpec(
    name="search_faq",
    description=(
        "Поиск в базе вопросов и ответов HR. Используй когда вопрос пользователя "
        "похож на типовой и может уже быть в FAQ: например, про график, отпуск, "
        "корпоратив, мероприятия, простые процедуры."
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

_SEARCH_DOCUMENTS_SPEC = ToolSpec(
    name="search_documents",
    description=(
        "Поиск во внутренних документах компании: регламентах, инструкциях, "
        "велкомбуке, бланках для заполнения. Используй когда нужны точные "
        "сведения из официальных документов или когда пользователь спрашивает "
        "про бланк заявления."
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

_SEARCH_WIKI_SPEC = ToolSpec(
    name="search_wiki",
    description=(
        "Поиск в базе знаний компании на внешнем сайте. Используй для "
        "поиска статей и описаний бизнес-процессов. Темы: электронный документооборот ЭДО EDI со СБИС и Корус,"
        "работа с маркируемыми товарами Честный знак, интеграции со сторонними сервисами и системами,"
        "например, по оплате фискализации, по работе с маркетплейсами"
        "инструкции для закупок по подключению поставщиков к ЭДО, инструкции, как заполнять в 1С карточки на товары."
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
                "description": "Имя и\или фамилия",
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
        "ответить ни одно из других средств, и этот вопрос касается внутренней работы компании"
        "и нужна помощь живого HR."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)


# ---------- Реестр ----------

TOOLS: dict[str, RegisteredTool] = {
    # Agent-internal: агент сам ищет, потом формирует ответ из контекста
    "search_faq": RegisteredTool("search_faq", "agent_internal", _SEARCH_FAQ_SPEC),
    "search_documents": RegisteredTool("search_documents", "agent_internal", _SEARCH_DOCUMENTS_SPEC),
    "search_wiki": RegisteredTool("search_wiki", "agent_internal", _SEARCH_WIKI_SPEC),

    # Bot-command: возвращаем боту, он сам выполнит
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