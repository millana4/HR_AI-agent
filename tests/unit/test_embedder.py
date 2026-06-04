"""Тесты embedder. Загружают реальную модель — медленно!"""
import pytest

from app.rag.embedder import VECTOR_SIZE, Embedder, get_embedder


@pytest.fixture(scope="module")
def embedder():
    """Один embedder на все тесты в модуле — модель грузится один раз."""
    return get_embedder()


async def test_embed_query_returns_vector(embedder: Embedder):
    """Запрос превращается в вектор нужной размерности."""
    vector = await embedder.embed_query("Сколько дней отпуска?")
    assert isinstance(vector, list)
    assert len(vector) == VECTOR_SIZE
    assert all(isinstance(x, float) for x in vector)


async def test_embed_documents_returns_list_of_vectors(embedder: Embedder):
    """Несколько документов — несколько векторов."""
    vectors = await embedder.embed_documents([
        "Стандартный отпуск 28 дней",
        "Больничный оформляется через отдел кадров",
    ])
    assert len(vectors) == 2
    assert all(len(v) == VECTOR_SIZE for v in vectors)


async def test_embed_documents_empty_list(embedder: Embedder):
    """Пустой список — пустой результат."""
    result = await embedder.embed_documents([])
    assert result == []


async def test_similar_texts_have_close_vectors(embedder: Embedder):
    """Семантически близкие тексты имеют косинусное сходство выше, чем далёкие."""
    vectors = await embedder.embed_documents([
        "Сколько дней отпуска положено?",
        "Какой период отдыха предусмотрен?",
        "Как работает корпоративный кафе?",
    ])

    # Скалярное произведение нормализованных векторов = косинусное сходство
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_close = cosine(vectors[0], vectors[1])  # про отпуск
    sim_far = cosine(vectors[0], vectors[2])  # отпуск vs кафе

    assert sim_close > sim_far


async def test_query_and_document_use_different_prefixes(embedder: Embedder):
    """Один и тот же текст как query и как document даёт разные векторы."""
    text = "Сколько дней отпуска?"
    query_vec = await embedder.embed_query(text)
    doc_vecs = await embedder.embed_documents([text])

    # Если префиксы работают, векторы должны различаться
    assert query_vec != doc_vecs[0]