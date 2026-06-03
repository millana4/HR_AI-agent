"""
Парсер запроса пользователя: маскирование имён и фамилий.

Использование:
    parser = PiiParser()
    await parser.ensure_ready(nocodb_client, correlation_id)
    result = parser.parse("Найди телефон Иванова")
    # result.masked_text = "Найди телефон [NAME]"
    # result.found_names = ["Иванов"]
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

# Плейсхолдер, которым заменяем имена и фамилии в тексте
NAME_PLACEHOLDER = "[NAME]"


@dataclass
class PiiParseResult:
    """Результат парсинга запроса."""

    masked_text: str
    """Текст с заменёнными именами/фамилиями на [NAME]."""

    found_names: list[str] = field(default_factory=list)
    """Список найденных имён/фамилий в именительном падеже, в порядке появления."""


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
                out_parts.append(NAME_PLACEHOLDER)
                found_names.append(lemma)
            else:
                out_parts.append(token)

        masked_text = "".join(out_parts)

        logger.debug(
            f"PII parser output: masked={masked_text!r}, found={found_names}",
            extra={"correlation_id": correlation_id},
        )

        return PiiParseResult(masked_text=masked_text, found_names=found_names)