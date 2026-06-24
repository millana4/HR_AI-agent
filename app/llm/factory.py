"""
Фабрика LLM-клиентов. Выбирает реализацию по Config.LLM_PROVIDER.
"""
from app.core.config import Config
from app.core.exceptions import ConfigError
from app.llm.base import BaseLLMClient
from app.llm.gigachat_client import GigaChatClient
from app.llm.yandex_client import YandexClient


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """
    Создать LLM-клиент по конфигу или явно указанному провайдеру.

    Args:
        provider: имя провайдера ("yandex" / "gigachat"). Если None —
            берётся Config.LLM_PROVIDER. Явный параметр нужен для fallback
            (шаг 7): создать запасной GigaChat-клиент независимо от конфига.
    """
    name = (provider or Config.LLM_PROVIDER).lower()

    if name == "yandex":
        return YandexClient()
    if name == "gigachat":
        return GigaChatClient()

    raise ConfigError(f"Unsupported LLM_PROVIDER: {name}")