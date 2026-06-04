"""
Разбивка длинного текста на чанки для эмбеддинга.

Стратегия — recursive: режем по приоритетным разделителям (абзацы → предложения → слова),
стараясь не рвать смысловые единицы.

Размер чанка считаем в словах (а не в символах/токенах) — это проще
и достаточно точно для русского текста.
"""
import re
from dataclasses import dataclass

from app.core.logging import get_logger


logger = get_logger(__name__)


# Разделители в порядке приоритета: чем выше — тем «крупнее» граница.
# Сначала пытаемся резать по двойным переводам строк (абзацы),
# потом по одинарным, потом по предложениям, в крайнем случае — по словам.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]


# Дефолтные параметры
DEFAULT_CHUNK_SIZE = 500   # слов
DEFAULT_CHUNK_OVERLAP = 50  # слов


@dataclass
class Chunk:
    """Один чанк текста с метаданными."""

    text: str
    index: int  # порядковый номер чанка в документе


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    correlation_id: str = "-",
) -> list[Chunk]:
    """
    Разбить текст на чанки.

    Args:
        text: исходный текст
        chunk_size: максимальный размер чанка в словах
        chunk_overlap: количество слов перекрытия между соседними чанками
        correlation_id: для трассировки

    Returns:
        Список объектов Chunk.
    """
    if not text or not text.strip():
        return []

    # Очистка лишних пробелов и переносов
    text = _normalize_whitespace(text)

    # Если текст короткий — возвращаем как один чанк
    word_count = len(text.split())
    if word_count <= chunk_size:
        logger.debug(
            f"Text is short ({word_count} words), returning as single chunk",
            extra={"correlation_id": correlation_id},
        )
        return [Chunk(text=text, index=0)]

    # Иначе — recursive split
    pieces = _recursive_split(text, chunk_size)

    # Склеиваем мелкие куски в чанки целевого размера с overlap
    chunks_text = _merge_with_overlap(pieces, chunk_size, chunk_overlap)

    chunks = [Chunk(text=t, index=i) for i, t in enumerate(chunks_text)]

    logger.debug(
        f"Split text of {word_count} words into {len(chunks)} chunks",
        extra={"correlation_id": correlation_id},
    )
    return chunks


def _normalize_whitespace(text: str) -> str:
    """Убрать лишние пробелы. Сохраняет двойные переводы строк (границы абзацев)."""
    # Несколько подряд пробелов/табов → один пробел
    text = re.sub(r"[ \t]+", " ", text)
    # Тройные и более переводы строк → двойные
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Пробелы вокруг переводов строк
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _recursive_split(text: str, chunk_size: int) -> list[str]:
    """
    Разбить текст на мелкие куски, рекурсивно понижая уровень разделителя.

    Возвращает список кусков, каждый из которых не превышает chunk_size слов.
    """
    if len(text.split()) <= chunk_size:
        return [text]

    for sep in _SEPARATORS:
        if sep not in text:
            continue

        parts = text.split(sep)
        result: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part.split()) <= chunk_size:
                result.append(part)
            else:
                # Кусок всё ещё большой — спускаемся на следующий уровень
                result.extend(_recursive_split(part, chunk_size))
        return result

    # Разделители не помогли (например, текст без пробелов) — режем по словам
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def _merge_with_overlap(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Склеить мелкие куски в чанки целевого размера с перекрытием.

    Каждый следующий чанк начинается на chunk_overlap слов раньше, чем заканчивается предыдущий.
    """
    chunks: list[str] = []
    current_words: list[str] = []

    for piece in pieces:
        piece_words = piece.split()

        if len(current_words) + len(piece_words) <= chunk_size:
            current_words.extend(piece_words)
        else:
            # Текущий чанк уже не вмещает следующий piece — сохраняем его
            if current_words:
                chunks.append(" ".join(current_words))

            # Начинаем новый чанк с overlap-хвостом предыдущего
            if chunk_overlap > 0 and current_words:
                overlap_words = current_words[-chunk_overlap:]
            else:
                overlap_words = []
            current_words = overlap_words + piece_words

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks