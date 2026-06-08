"""
Интеграционные тесты эндпоинта /api/health.

Health должен работать без авторизации и возвращать correlation_id
в заголовке через CorrelationIdMiddleware.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app


@pytest.fixture
def client():
    """TestClient без подъёма lifespan. agent_loop заглушаем, чтобы не
    падало при импорте роутов."""
    app.state.agent_loop = MagicMock()
    return TestClient(app)


def test_health_without_api_key_returns_200(client: TestClient):
    """Health должен работать без авторизации."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_health_returns_correlation_id_header(client: TestClient):
    response = client.get("/api/health")
    assert "x-correlation-id" in response.headers
    assert len(response.headers["x-correlation-id"]) == 16