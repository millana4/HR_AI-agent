"""Тесты фабрики LLM-клиентов."""
from unittest.mock import patch

import pytest

from app.core.exceptions import ConfigError
from app.llm.factory import get_llm_client
from app.llm.gigachat import GigaChatClient


def test_factory_returns_gigachat_client():
    with patch("app.llm.factory.Config.LLM_PROVIDER", "gigachat"):
        client = get_llm_client()
        assert isinstance(client, GigaChatClient)


def test_factory_case_insensitive():
    with patch("app.llm.factory.Config.LLM_PROVIDER", "GigaChat"):
        client = get_llm_client()
        assert isinstance(client, GigaChatClient)


def test_factory_unknown_provider_raises():
    with patch("app.llm.factory.Config.LLM_PROVIDER", "unknown"):
        with pytest.raises(ConfigError) as exc_info:
            get_llm_client()
        assert "unknown" in str(exc_info.value).lower()