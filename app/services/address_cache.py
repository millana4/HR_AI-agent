"""
Кеш словаря адресов магазинов и аптек для каскадной классификации.

Назначение (Дизайн А): tool search_shop / search_drugstore выбирает LLM на
Pass 1. После этого из запроса пользователя извлекается ТОПОНИМ (улица в
приоритете, иначе город) и кладётся в query для бота — чтобы передать точное
название из справочника, а не то, что LLM коряво вычленила.

Словарь строится раз в день из NocoDB (SHOP_TABLE_ID, DRUGSTORE_TABLE_ID),
поле Title парсится на город/улицу. Структура:
    lowercase словоформа → (нормальная форма, тип "street"|"city")

Источник адреса — поле Title формата:
    "Магазин ГОРОД, УЛИЦА ДОМ"  или  "Магазин УЛИЦА ДОМ"  (запятая = есть город)
"""
import re
from dataclasses import dataclass
from datetime import date

import pymorphy3

from app.core.config import Config
from app.core.logging import get_logger
from app.repositories.nocodb_client import NocoDBClient


logger = get_logger(__name__)


@dataclass(frozen=True)
class AddressEntry:
    """Запись словаря: нормальная форма топонима и его тип."""
    normal: str          # «Строителей», «Кудрово»
    kind: str            # "street" | "city"


# Префиксы и маркеры, которые выкидываем из Title.
_PREFIX_RE = re.compile(r"^\*?\s*(магазин|аптека)\s+", re.IGNORECASE)
# Маркеры дом/корпус/улица — целиком слова.
_MARKER_RE = re.compile(
    r"\b(ул|улица|д|дом|корп|кор|к|стр|литера|лит)\b\.?",
    re.IGNORECASE,
)
# Номер дома с возможной буквой/корпусом: «1A», «143», «16к3», «69г».
_PURE_NUM_RE = re.compile(r"^\d+[а-яёa-z]?$", re.IGNORECASE)

# Разрезать слитные «Софийская57» → «Софийская», «57».
_SPLIT_ALNUM_RE = re.compile(r"([а-яёА-ЯЁ]+)(\d+)")
# Чисто числовой токен (номер дома) или короткий мусор.
_PURE_NUM_RE = re.compile(r"^\d+[а-яё]?$", re.IGNORECASE)


class AddressCache:
    """Кеш словоформ адресов магазинов и аптек."""

    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()
        self._forms: dict[str, AddressEntry] = {}
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
        logger.info("Building address cache", extra={"correlation_id": correlation_id})
        await self._rebuild(client, correlation_id)
        self._built_for_date = today

    def get_forms(self) -> dict[str, AddressEntry]:
        """Словарь словоформ. Не вызывать без ensure_fresh()."""
        return self._forms

    def extract_address(self, text: str) -> str | None:
        """
        Извлечь топоним из запроса пользователя.

        Возвращает нормальную форму УЛИЦЫ (приоритет) или ГОРОДА, если улицы
        в запросе нет. None — если ничего не нашлось (тогда фоллбэк на query
        от LLM в вызывающем коде).
        """
        if not self._forms:
            return None
        tokens = self._tokenize(text)

        found_street: str | None = None
        found_city: str | None = None
        for tok in tokens:
            entry = self._lookup(tok)
            if entry is None:
                continue
            if entry.kind == "street" and found_street is None:
                found_street = entry.normal
            elif entry.kind == "city" and found_city is None:
                found_city = entry.normal

        # Улица в приоритете, иначе город.
        return found_street or found_city

    # ---------- построение ----------

    async def _rebuild(self, client: NocoDBClient, correlation_id: str) -> None:
        titles: list[str] = []
        for table_id in (Config.SHOP_TABLE_ID, Config.DRUGSTORE_TABLE_ID):
            if not table_id:
                continue
            records = await client.list_records(table_id, correlation_id=correlation_id)
            for rec in records:
                title = (rec.get("Title") or "").strip()
                if title:
                    titles.append(title)

        new_forms: dict[str, AddressEntry] = {}
        for title in titles:
            city, street_tokens = self._parse_title(title)
            # Город (может быть из нескольких слов — кладём по словам и целиком).
            if city:
                self._add_topo(city, "city", new_forms)
            # Улица: каждое слово отдельно + пара целиком (для составных улиц).
            self._add_street(street_tokens, new_forms)

        self._forms = new_forms
        logger.info(
            f"Address cache rebuilt: {len(titles)} titles, {len(new_forms)} word forms",
            extra={"correlation_id": correlation_id},
        )

    def _parse_title(self, title: str) -> tuple[str | None, list[str]]:
        """
        Разобрать Title на (город | None, [слова улицы]).

        Если есть запятая — слева город, справа улица. Иначе город=None.
        Маркеры «ул.»/«д.» убираем ДО деления, чтобы «ул., д.» не приняли
        за разделитель город/улица.
        """
        # Убираем префикс «Магазин»/«Аптека»/«*».
        body = _PREFIX_RE.sub("", title).strip()
        # Убираем маркеры ул./д./корп. заранее (вместе с точкой после них).
        body = _MARKER_RE.sub(" ", body)
        # Схлопываем образовавшиеся «, » без содержимого и лишние пробелы.
        body = re.sub(r"\s+", " ", body).strip()

        city_part: str | None = None
        street_part: str = body
        if "," in body:
            left, right = body.split(",", 1)
            city_part = left.strip(" ,") or None
            street_part = right.strip(" ,")

        street_tokens = self._clean_tokens(street_part)
        return city_part, street_tokens

    def _clean_tokens(self, part: str) -> list[str]:
        """Очистить часть адреса: убрать маркеры, номера домов, разрезать слитное."""
        # Разрезаем слитные «Софийская57» → «Софийская 57».
        part = _SPLIT_ALNUM_RE.sub(r"\1 \2", part)
        # Убираем маркеры ул./д./к.
        part = _MARKER_RE.sub(" ", part)
        result: list[str] = []
        for raw in part.split():
            tok = raw.strip(".,*").strip()
            if not tok:
                continue
            if _PURE_NUM_RE.match(tok):  # номер дома
                continue
            if len(tok) <= 1:
                continue
            result.append(tok)
        return result

    def _add_topo(self, phrase: str, kind: str, forms: dict[str, AddressEntry]) -> None:
        """Добавить топоним (город или улицу) из фразы: по словам и целиком."""
        tokens = self._clean_tokens(phrase)
        if not tokens:
            return
        # Целая фраза (для «Великий Новгород», «Авиаторов Балтики»).
        whole = " ".join(tokens)
        normal_whole = whole.capitalize()
        forms[whole.lower()] = AddressEntry(normal=normal_whole, kind=kind)
        # По отдельным словам.
        for tok in tokens:
            self._expand_token(tok, kind, normal_whole if len(tokens) == 1 else tok.capitalize(), forms)

    def _add_street(self, tokens: list[str], forms: dict[str, AddressEntry]) -> None:
        """Добавить улицу: слова по отдельности + пара целиком (составные улицы)."""
        if not tokens:
            return
        # Составная улица целиком («Авиаторов Балтики»).
        if len(tokens) >= 2:
            whole = " ".join(tokens)
            forms[whole.lower()] = AddressEntry(normal=whole.title(), kind="street")
        for tok in tokens:
            self._expand_token(tok, "street", tok.capitalize(), forms)

    def _expand_token(
        self,
        token: str,
        kind: str,
        normal: str,
        forms: dict[str, AddressEntry],
    ) -> None:
        """Добавить токен и его словоформы (pymorphy) в словарь."""
        forms.setdefault(token.lower(), AddressEntry(normal=normal, kind=kind))
        try:
            parsed = self._morph.parse(token)
        except Exception:
            return
        for variant in parsed[:1]:  # только первый разбор, чтобы не плодить шум
            try:
                lexeme = variant.lexeme
            except Exception:
                continue
            for form in lexeme:
                w = form.word.lower()
                if w and w not in forms:
                    forms[w] = AddressEntry(normal=normal, kind=kind)

    # ---------- разбор запроса ----------

    def _tokenize(self, text: str) -> list[str]:
        """Разбить запрос на слова + добавить пары соседних слов (составные улицы)."""
        words = [w.strip(".,*").lower() for w in text.split() if w.strip(".,*")]
        result = list(words)
        # Пары соседних слов — чтобы поймать «авиаторов балтики».
        for i in range(len(words) - 1):
            result.append(f"{words[i]} {words[i + 1]}")
        return result

    def _lookup(self, token: str) -> AddressEntry | None:
        """Найти токен в словаре: точно или по нормальной форме."""
        entry = self._forms.get(token)
        if entry:
            return entry
        # Нормализуем через pymorphy и пробуем снова.
        try:
            parsed = self._morph.parse(token)
            if parsed:
                norm = parsed[0].normal_form
                return self._forms.get(norm)
        except Exception:
            pass
        return None


# Глобальный singleton.
_address_cache = AddressCache()


def get_address_cache() -> AddressCache:
    """Получить глобальный экземпляр кеша адресов."""
    return _address_cache