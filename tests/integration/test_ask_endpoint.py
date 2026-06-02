import pytest
from fastapi.testclient import TestClient

from app.core.config import Config
from app.main import app


@pytest.fixture
def client() -> TestClient:
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
# Валидация запроса
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


# ============================================
# Успешный сценарий (заглушка)
# ============================================

def test_ask_minimal_valid_request(client: TestClient, auth_headers: dict):
    response = client.post(
        "/api/ask",
        headers=auth_headers,
        json={"user_id": 123, "request": "Где найти бланк отпуска?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["response_type"] == "text"
    assert body["tool_used"] == "answer_general"
    assert "correlation_id" in body
    assert len(body["correlation_id"]) == 16


def test_ask_passes_through_correlation_id_from_body(
    client: TestClient, auth_headers: dict
):
    """Если correlation_id передан в теле — он же возвращается в ответе."""
    response = client.post(
        "/api/ask",
        headers=auth_headers,
        json={
            "user_id": 123,
            "request": "Привет",
            "correlation_id": "my-test-id-12345",
        },
    )
    assert response.status_code == 200
    assert response.json()["correlation_id"] == "my-test-id-12345"


def test_ask_passes_through_correlation_id_from_header(
    client: TestClient, auth_headers: dict
):
    """Если correlation_id передан в заголовке — он же возвращается."""
    headers = {**auth_headers, "X-Correlation-Id": "header-cid-9876"}
    response = client.post(
        "/api/ask",
        headers=headers,
        json={"user_id": 123, "request": "Привет"},
    )
    assert response.status_code == 200
    assert response.json()["correlation_id"] == "header-cid-9876"
    assert response.headers["x-correlation-id"] == "header-cid-9876"


# ============================================
# Эндпоинт /api/health
# ============================================

def test_health_without_api_key_returns_200(client: TestClient):
    """Health должен работать без авторизации."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_health_returns_correlation_id_header(client: TestClient):
    response = client.get("/api/health")
    assert "x-correlation-id" in response.headers
    assert len(response.headers["x-correlation-id"]) == 16