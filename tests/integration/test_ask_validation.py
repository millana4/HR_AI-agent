"""
Интеграционные тесты валидации и авторизации эндпоинта /api/ask.

Все тесты здесь проверяют поведение ДО того, как запрос дойдёт до
agent_loop: middleware (auth, correlation_id) и валидация pydantic-схемы.

Тесты успешной обработки запросов с моками LLM/SessionStore/PiiParser
живут в шаге 12.4 (test_agent_loop.py).
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.core.config import Config
from app.main import app


@pytest.fixture
def client():
    """
    TestClient с заглушкой agent_loop в app.state.

    Lifespan в тестах не запускается, поэтому реальный AgentLoop туда
    не попадёт. Для тестов валидации/авторизации этого достаточно:
    запросы здесь падают раньше, чем кто-либо обратится к agent_loop.
    """
    app.state.agent_loop = MagicMock()
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict:
    return {"X-API-Key": Config.AI_AGENT_API_KEY}


# ============================================
# Авторизация
# ============================================

def test_ask_without_api_key_returns_401(client: TestClient):
    response = client.post(
        "/api/ask",
        json={"user_id": 123, "request": "Привет"},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"] == "Missing X-API-Key"
    assert "correlation_id" in body


def test_ask_with_wrong_api_key_returns_401(client: TestClient):
    response = client.post(
        "/api/ask",
        headers={"X-API-Key": "wrong_key"},
        json={"user_id": 123, "request": "Привет"},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"] == "Invalid X-API-Key"


# ============================================
# Валидация тела запроса
# ============================================

def test_ask_missing_user_id_returns_400(client: TestClient, auth_headers: dict):
    response = client.post(
        "/api/ask",
        headers=auth_headers,
        json={"request": "Привет"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "Validation error"
    assert "user_id" in body["detail"]


def test_ask_missing_request_returns_400(client: TestClient, auth_headers: dict):
    response = client.post(
        "/api/ask",
        headers=auth_headers,
        json={"user_id": 123},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "Validation error"
    assert "request" in body["detail"]


def test_ask_empty_request_returns_400(client: TestClient, auth_headers: dict):
    response = client.post(
        "/api/ask",
        headers=auth_headers,
        json={"user_id": 123, "request": ""},
    )
    assert response.status_code == 400


def test_ask_too_long_request_returns_400(client: TestClient, auth_headers: dict):
    response = client.post(
        "/api/ask",
        headers=auth_headers,
        json={"user_id": 123, "request": "a" * 2001},
    )
    assert response.status_code == 400