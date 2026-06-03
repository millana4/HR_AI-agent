"""
Фабрика LLM-клиентов. Выбирает реализацию по Config.LLM_PROVIDER.
"""
from app.core.config import Config
from app.core.exceptions import ConfigError
from app.llm.base import BaseLLMClient
from app.llm.gigachat import GigaChatClient


def get_llm_client() -> BaseLLMClient:
    """
    Создать LLM-клиент по конфигу.

    Сейчас поддерживается только gigachat. Когда появится другой провайдер
    (например, minimax) — добавится новая ветка if/elif.
    """
    provider = Config.LLM_PROVIDER.lower()

    if provider == "gigachat":
        return GigaChatClient()

    raise ConfigError(f"Unsupported LLM_PROVIDER: {Config.LLM_PROVIDER}")