"""
Кеш словаря ФИО для PII-парсера.

Словарь строится раз в день (при первом запросе после полуночи):
- Тянем ФИО сотрудников из NocoDB
- Отбрасываем отчество
- Раскрываем имена и фамилии во все словоформы через pymorphy3
- Кешируем в памяти процесса до конца суток

Структура кеша: lowercase словоформа → лемма в именительном падеже
"""
from datetime import date

import pymorphy3

from app.core.logging import get_logger
from app.repositories.nocodb_client import NocoDBClient
from app.repositories.pivot import fetch_pivot


logger = get_logger(__name__)


class PiiCache:
    """
    Кеш словоформ ФИО сотрудников.

    Использование:
        cache = PiiCache()
        await cache.ensure_fresh(nocodb_client, correlation_id)
        forms = cache.get_forms()  # dict: lowercase форма → лемма
    """

    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()
        self._forms: dict[str, str] = {}
        self._built_for_date: date | None = None

    async def ensure_fresh(
        self,
        client: NocoDBClient,
        correlation_id: str = "-",
    ) -> None:
        """Если кеш ещё не построен сегодня — построить."""
        today = date.today()
        if self._built_for_date == today and self._forms:
            return

        logger.info(
            "Building PII cache",
            extra={"correlation_id": correlation_id},
        )
        await self._rebuild(client, correlation_id)
        self._built_for_date = today

    def get_forms(self) -> dict[str, str]:
        """Получить словарь словоформ. Не вызывать без ensure_fresh()."""
        return self._forms

    async def _rebuild(
        self,
        client: NocoDBClient,
        correlation_id: str,
    ) -> None:
        fios = await fetch_pivot(client, correlation_id=correlation_id)

        # Собираем уникальные имена и фамилии (без отчества)
        name_tokens: set[str] = set()
        for fio in fios:
            tokens = fio.split()
            if len(tokens) >= 3:
                # Иванов Иван Иванович → отбрасываем отчество
                tokens = tokens[:-1]
            # для строк из 2 слов оставляем оба
            # для строк из 1 слова — оставляем
            for token in tokens:
                clean = token.strip()
                if clean:
                    name_tokens.add(clean)

        logger.debug(
            f"PII cache: {len(name_tokens)} unique name tokens to expand",
            extra={"correlation_id": correlation_id},
        )

        # Раскрываем во все словоформы
        new_forms: dict[str, str] = {}
        for token in name_tokens:
            self._expand_token(token, new_forms)

        self._forms = new_forms

        logger.info(
            f"PII cache rebuilt: {len(name_tokens)} lemmas, "
            f"{len(new_forms)} word forms",
            extra={"correlation_id": correlation_id},
        )

    def _expand_token(self, token: str, forms_dict: dict[str, str]) -> None:
        """
        Получить все словоформы токена и записать в forms_dict.

        Ключ — словоформа в lowercase. Значение — лемма с заглавной буквы.
        """
        parsed = self._morph.parse(token)
        if not parsed:
            # pymorphy не разобрал — кладём сам токен
            forms_dict[token.lower()] = token
            return

        # Лемма в именительном падеже — берём из первого варианта разбора
        lemma = parsed[0].normal_form.capitalize()

        # Сначала кладём сам исходный токен в обоих регистрах
        forms_dict[token.lower()] = lemma

        # Все словоформы из всех вариантов разбора
        for variant in parsed:
            try:
                lexeme = variant.lexeme
            except Exception:
                continue
            for form in lexeme:
                word = form.word.lower()
                if word and word not in forms_dict:
                    forms_dict[word] = lemma


# Глобальный singleton кеша
_pii_cache = PiiCache()


def get_pii_cache() -> PiiCache:
    """Получить глобальный экземпляр кеша."""
    return _pii_cache