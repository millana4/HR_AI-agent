# HR AI Agent

ИИ-агент для HR-бота МАВИС-гид. Отдельный сервис, отвечающий на открытые вопросы сотрудников.

## Архитектура

- FastAPI-сервис на отдельном сервере
- Внешняя LLM (MiniMax / GigaChat / другая)
- RAG: Qdrant + multilingual-e5-large
- Источники данных: NocoDB (FAQ, история), Selectel CDN (документы), std.kitdev.ru (база знаний)

## Стек

Python 3.12, FastAPI, Qdrant, Redis, Natasha, sentence-transformers.

## Redis
docker start hr-ai-redis

docker ps | grep hr-ai-redis

Проверка:

docker exec hr-ai-redis redis-cli ping

## Qdrant
docker start hr-ai-qdrant

curl http://localhost:6333/collections

Проверка:
docker ps | grep hr-ai-qdrant