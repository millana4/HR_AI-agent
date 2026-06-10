import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / "devops" / ".env")


class Config:
    # Безопасность
    AI_AGENT_API_KEY = os.getenv("AI_AGENT_API_KEY")
    SESSION_HASH_SALT = os.getenv("SESSION_HASH_SALT")

    # Логирование
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/agent.log")
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 3))

    # NocoDB
    NOCODB_SERVER = os.getenv("NOCODB_SERVER")
    NOCODB_API_TOKEN = os.getenv("NOCODB_API_TOKEN")

    AI_FAQ_TABLE_ID = os.getenv("AI_FAQ_TABLE_ID")
    AI_DOCUMENTS_TABLE_ID = os.getenv("AI_DOCUMENTS_TABLE_ID")
    AI_ANALYTICS_TABLE_ID = os.getenv("AI_ANALYTICS_TABLE_ID")
    PIVOT_TABLE_ID = os.getenv("PIVOT_TABLE_ID")

    ATS_MAVIS_BOOK_ID = os.getenv("ATS_MAVIS_BOOK_ID")
    ATS_VOTONIA_BOOK_ID = os.getenv("ATS_VOTONIA_BOOK_ID")

    # LLM
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))

    # GigaChat
    GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
    GIGACHAT_AUTH_URL = os.getenv("GIGACHAT_AUTH_URL")
    GIGACHAT_API_URL = os.getenv("GIGACHAT_API_URL")
    GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", 24))
    SESSION_MAX_MESSAGES = int(os.getenv("SESSION_MAX_MESSAGES", 10))

    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge")