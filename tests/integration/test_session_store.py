"""
Интеграционные тесты SessionStore.
Требуют запущенный Redis на localhost:6379.
"""
from datetime import date, timedelta

import pytest
import pytest_asyncio

from app.services.session_store import SessionStore, make_session_key




@pytest_asyncio.fixture
async def store():
    """Подключённый SessionStore с очисткой тестовых данных."""
    s = SessionStore()
    await s.connect()
    yield s
    # Очистка после теста — удаляем всё, что начинается с session:test_*
    client = s._get_client()
    keys = await client.keys("session:99999*")
    if keys:
        await client.delete(*keys)
    await s.disconnect()


# ============================================
# make_session_key
# ============================================

def test_make_session_key_format():
    key = make_session_key(123, date(2026, 6, 2))
    assert key == "session:123:2026-06-02"


def test_make_session_key_default_today():
    key = make_session_key(123)
    today = date.today().isoformat()
    assert key == f"session:123:{today}"


# ============================================
# append + get_history
# ============================================

async def test_append_and_get_single_message(store: SessionStore):
    user_id = 99999
    await store.append(user_id=user_id, role="user", content="Привет")
    history = await store.get_history(user_id=user_id)
    assert history == [{"role": "user", "content": "Привет"}]


async def test_append_multiple_messages(store: SessionStore):
    user_id = 99999
    await store.append(user_id=user_id, role="user", content="Вопрос 1")
    await store.append(user_id=user_id, role="assistant", content="Ответ 1")
    await store.append(user_id=user_id, role="user", content="Вопрос 2")
    history = await store.get_history(user_id=user_id)
    assert len(history) == 3
    assert history[0]["content"] == "Вопрос 1"
    assert history[2]["content"] == "Вопрос 2"


async def test_get_history_empty(store: SessionStore):
    user_id = 99999
    history = await store.get_history(user_id=user_id)
    assert history == []


# ============================================
# Изоляция сессий
# ============================================

async def test_different_users_isolated(store: SessionStore):
    await store.append(user_id=99999, role="user", content="User A")
    await store.append(user_id=99998, role="user", content="User B")
    history_a = await store.get_history(user_id=99999)
    history_b = await store.get_history(user_id=99998)
    assert history_a == [{"role": "user", "content": "User A"}]
    assert history_b == [{"role": "user", "content": "User B"}]
    # Уберём за вторым юзером
    await store.clear(user_id=99998)


async def test_different_dates_isolated(store: SessionStore):
    """Сообщения от разных дат хранятся в разных ключах."""
    user_id = 99999
    yesterday = date.today() - timedelta(days=1)

    await store.append(
        user_id=user_id,
        role="user",
        content="Yesterday",
        target_date=yesterday,
    )
    await store.append(
        user_id=user_id,
        role="user",
        content="Today",
        target_date=date.today(),
    )

    history_today = await store.get_history(user_id=user_id, target_date=date.today())
    assert any(m["content"] == "Today" for m in history_today)
    assert not any(m["content"] == "Yesterday" for m in history_today)

    history_yesterday = await store.get_history(user_id=user_id, target_date=yesterday)
    assert any(m["content"] == "Yesterday" for m in history_yesterday)
    assert not any(m["content"] == "Today" for m in history_yesterday)

    # Уборка за вчерашний ключ — фикстура чистит только session:99999*, но это покрывает обоих
    await store.clear(user_id=user_id, target_date=yesterday)


# ============================================
# Ограничение размера
# ============================================

async def test_max_messages_truncation(store: SessionStore):
    """Должно храниться не больше SESSION_MAX_MESSAGES пар (2*N сообщений)."""
    from app.core.config import Config
    max_pairs = Config.SESSION_MAX_MESSAGES
    user_id = 99999

    # Пишем больше, чем лимит
    total = (max_pairs + 5) * 2
    for i in range(total):
        await store.append(
            user_id=user_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"msg {i}",
        )

    history = await store.get_history(user_id=user_id)
    assert len(history) == max_pairs * 2
    # Старые сообщения вытеснены — должен остаться хвост
    assert history[-1]["content"] == f"msg {total - 1}"


# ============================================
# Clear
# ============================================

async def test_clear_removes_history(store: SessionStore):
    user_id = 99999
    await store.append(user_id=user_id, role="user", content="Привет")
    await store.clear(user_id=user_id)
    history = await store.get_history(user_id=user_id)
    assert history == []