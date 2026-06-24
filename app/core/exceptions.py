"""
Кастомные исключения приложения. Используются для отделения доменных
ошибок от технических (httpx, redis и т.д.).
"""


class AgentError(Exception):
    """Базовое исключение приложения."""


class ConfigError(AgentError):
    """Ошибка конфигурации."""


class LLMError(AgentError):
    """Ошибка взаимодействия с LLM."""


class LLMTimeoutError(LLMError):
    """LLM не ответила в отведённое время."""


class LLMAuthError(LLMError):
    """
    LLM отвергла запрос по авторизации/оплате (HTTP 401/403).

    Для Yandex это сигнал «кончились деньги или невалидный ключ» — повод
    для алерта администраторам (в отличие от временных сбоев).
    """


class ToolExecutionError(AgentError):
    """Ошибка выполнения tool."""


class RepositoryError(AgentError):
    """Ошибка работы с внешним хранилищем (NocoDB, Qdrant, Redis)."""


class AuthError(AgentError):
    """Ошибка авторизации."""