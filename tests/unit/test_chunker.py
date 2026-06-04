"""Тесты чанкера."""
from app.rag.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    split_text,
)


def test_empty_text_returns_empty_list():
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text_returns_single_chunk():
    text = "Короткий текст про отпуск."
    chunks = split_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].index == 0


def test_long_text_split_into_multiple_chunks():
    """Текст из 1500 слов должен разбиться на несколько чанков."""
    text = " ".join(["слово"] * 1500)
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 3
    # Каждый чанк не превышает chunk_size (за исключением overlap)
    for chunk in chunks:
        assert len(chunk.text.split()) <= 500 + 50


def test_chunks_have_overlap():
    """Соседние чанки должны иметь общие слова."""
    # Текст из уникальных слов, чтобы было видно overlap
    words = [f"слово{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)

    # Проверим, что между чанком 0 и чанком 1 есть общие слова
    chunk0_words = set(chunks[0].text.split())
    chunk1_words = set(chunks[1].text.split())
    common = chunk0_words & chunk1_words
    assert len(common) > 0, "Соседние чанки должны иметь перекрытие"


def test_respects_paragraph_boundaries():
    """Чанкер должен предпочитать резать по абзацам."""
    para1 = " ".join(["абзацодин"] * 200)
    para2 = " ".join(["абзацдва"] * 200)
    para3 = " ".join(["абзацтри"] * 200)
    text = f"{para1}\n\n{para2}\n\n{para3}"

    chunks = split_text(text, chunk_size=250, chunk_overlap=20)
    # Должно быть как минимум 3 чанка (по одному на абзац)
    assert len(chunks) >= 3


def test_chunks_indices_sequential():
    """Индексы чанков должны идти 0, 1, 2..."""
    text = " ".join(["слово"] * 2000)
    chunks = split_text(text, chunk_size=300, chunk_overlap=30)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_normalizes_whitespace():
    """Множественные пробелы и переводы строк должны схлопываться."""
    text = "Текст   с    лишними    пробелами\n\n\n\nи переводами."
    chunks = split_text(text)
    assert "   " not in chunks[0].text
    assert "\n\n\n" not in chunks[0].text


def test_long_paragraph_split_by_sentences():
    """Один длинный абзац должен резаться по предложениям."""
    sentence = " ".join(["слово"] * 100) + "."
    text = " ".join([sentence] * 10)  # 10 предложений по 101 слову

    chunks = split_text(text, chunk_size=300, chunk_overlap=20)
    assert len(chunks) >= 3