"""
Парсер запроса пользователя: маскирование имён и фамилий.

Использование:
    parser = PiiParser()
    await parser.ensure_ready(nocodb_client, correlation_id)
    result = parser.parse("Найди телефон Иванова")
    # result.masked_text = "Найди телефон NAME_1"
    # result.found_names = ["Иванов"]

Плейсхолдеры нумерованные (NAME_1, NAME_2, ...): такой формат не содержит
спецсимволов, поэтому внешняя LLM его не искажает (в отличие от [NAME],
у которого модель роняла скобку).
"""
import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.repositories.nocodb_client import NocoDBClient
from app.services.pii_cache import get_pii_cache


logger = get_logger(__name__)


# Регулярка для разбиения текста на токены и не-токены (пробелы/пунктуация).
# Группа 1: подряд буквы/цифры (включая дефис внутри: «Иванов-Сидоров»)
# Группа 2: всё остальное (пробелы, знаки препинания)
_TOKEN_PATTERN = re.compile(r"([\w-]+)|([^\w-]+)", re.UNICODE)

# Префикс плейсхолдера. К нему добавляется порядковый номер: NAME_1, NAME_2, ...
NAME_PLACEHOLDER_PREFIX = "NAME_"


def make_placeholder(index: int) -> str:
    """Сформировать плейсхолдер по номеру: 1 → 'NAME_1'."""
    return f"{NAME_PLACEHOLDER_PREFIX}{index}"


def mask_for_logs(name: str) -> str:
    """
    Маскирование для логов и аналитики: первые 2 буквы + звёздочки.

    'Иванов' → 'Ив****', 'Ян' → 'Ян', 'А' → 'А'.
    Необратимо — используется только для чтения человеком, нигде обратно
    не подставляется.
    """
    if len(name) <= 2:
        return name
    return name[:2] + "*" * (len(name) - 2)


@dataclass
class PiiParseResult:
    """Результат парсинга запроса."""

    masked_text: str
    """Текст с заменёнными именами/фамилиями на NAME_1, NAME_2, ..."""

    found_names: list[str] = field(default_factory=list)
    """Список найденных имён/фамилий в именительном падеже, в порядке появления.

    Индекс в списке + 1 соответствует номеру плейсхолдера: found_names[0] → NAME_1.
    """


class PiiParser:
    """
    Парсер запроса для маскирования ФИО.

    Перед использованием обязательно вызвать ensure_ready() — он подтянет
    или обновит кеш словоформ.
    """

    def __init__(self) -> None:
        self._cache = get_pii_cache()

    async def ensure_ready(
        self,
        client: NocoDBClient,
        correlation_id: str = "-",
    ) -> None:
        """Подготовить кеш словоформ. Безопасно вызывать на каждый запрос."""
        await self._cache.ensure_fresh(client, correlation_id=correlation_id)

    def parse(self, text: str, correlation_id: str = "-") -> PiiParseResult:
        """
        Распарсить запрос пользователя.

        Каждое найденное имя/фамилия заменяется на нумерованный плейсхолдер
        NAME_1, NAME_2, ... в порядке появления.

        Args:
            text: исходный текст запроса
            correlation_id: для трассировки

        Returns:
            PiiParseResult с маскированным текстом и списком найденных лемм.
        """
        logger.debug(
            f"PII parser input: {text!r}",
            extra={"correlation_id": correlation_id},
        )

        forms = self._cache.get_forms()
        if not forms:
            logger.warning(
                "PII cache is empty — returning text as is",
                extra={"correlation_id": correlation_id},
            )
            return PiiParseResult(masked_text=text)

        found_names: list[str] = []
        out_parts: list[str] = []

        for match in _TOKEN_PATTERN.finditer(text):
            token = match.group(1)
            separator = match.group(2)

            if separator is not None:
                out_parts.append(separator)
                continue

            # token — это слово
            lemma = forms.get(token.lower())
            if lemma is not None:
                found_names.append(lemma)
                out_parts.append(make_placeholder(len(found_names)))
            else:
                out_parts.append(token)

        masked_text = "".join(out_parts)

        # В лог пишем маскированные имена (2 буквы + звёздочки), не реальные.
        logged_names = [mask_for_logs(n) for n in found_names]
        logger.debug(
            f"PII parser output: masked={masked_text!r}, found={logged_names}",
            extra={"correlation_id": correlation_id},
        )

        return PiiParseResult(masked_text=masked_text, found_names=found_names)