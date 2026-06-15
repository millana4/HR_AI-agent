# HR AI Agent

ИИ-агент для HR-бота МАВИС-гид. Отдельный сервис, отвечающий на
открытые вопросы сотрудников по корпоративным данным. Бот обращается к нему
по HTTP.

## Что умеет

На каждый вопрос агент по двухпроходной схеме решает, как ответить:

- **Поиск во внутренних источниках** (`search_internal`) — ищет по подготовленной
  таблице FAQ, документам (регламенты, бланки) и базе знаний. Ответ
  может быть в любом из них, агент ищет везде одним запросом.
- **Команда боту** (`search_contacts`, `search_ats_mavis`, `search_ats_votonia`,
  `search_shop`, `search_drugstore`) — агент не отвечает сам, а возвращает боту
  инструкцию выполнить поиск по справочнику.
- **Общий ответ** (`answer_general`) — если вопрос не про работу в компании.
- **Форма HR** (`suggest_hr_form`) — если по корпоративному вопросу ничего
  не нашлось, агент предлагает обратиться к живому HR.

Для выбора отдела при поиске телефонов агент подгружает списки отделов
Мавис/Вотоня и кеширует их в Redis на сутки. Все текстовые ответы
и вызовы инструментов записываются в таблицу аналитики
(в обезличенном виде) — для последующего анализа HR.

## Защита персональных данных

Фамилии и имена из запроса маскируются (NAME_1, NAME_2, ...) до отправки во
внешнюю LLM. Реальные имена восстанавливаются только в финальном ответе
пользователю и в аргументах команд для бота. В истории сессий и в
аналитике хранятся только маскированные версии. В логи реальные имена не
пишутся — только маскированные.

## Архитектура

- FastAPI-сервис с эндпоинтами `POST /api/ask` + `/api/health`
- Внешняя LLM: GigaChat (модели Max → Pro → Lite с fallback по квоте)
- RAG: Qdrant + локальная модель multilingual-e5-large
- Источники данных: NocoDB — FAQ и метаданные документов, файлы в CDN
- Контракт бот ↔ агент: `docs/openapi.yaml`

## Стек

Python 3.12, FastAPI, Qdrant, Redis, pymorphy3, sentence-transformers.

## Настройка

Создать `devops/.env` из примера и заполнить значениями (ключи NocoDB,
GigaChat, общий API-ключ, соль, ID таблиц):

```commandline
cp devops/.env.example devops/.env
```

## Запуск

Поднять стек (агент + Qdrant + Redis):

```commandline
cd devops
docker compose up -d --build
```

Первый старт занимает несколько минут — скачивается модель эмбеддингов (~2 ГБ).
Проверка готовности:

```commandline
curl http://localhost:8000/api/health
```

Проиндексировать источники данных в Qdrant:

```commandline
docker compose run --rm indexer --faq --documents
```

Индексатор инкрементальный — повторный запуск переиндексирует только
изменённые записи.

## Проверка через Swagger

Откройте `http://localhost:8000/docs`, авторизуйтесь, затем `POST /api/ask`.

## Тесты

При поднятых Qdrant и Redis — все тесты:

```commandline
pytest
```

Без поднятых сервисов часть интеграционных тестов упадёт по причине отсутствия
подключения (это не ошибки кода).

## Структура

```
app/
├── core/           config, logging, security, exceptions
├── api/            FastAPI routes, middleware, error handlers, схемы
├── services/       agent_loop, pii_parser, pii_cache, session_store, departments_cache
├── llm/            BaseLLMClient, GigaChatClient, factory, промпты
├── tools/          registry, tools_internal (поиск в Qdrant)
├── rag/            qdrant_store, embedder, chunker
├── repositories/   nocodb_client, faq, documents, pivot, analytics
└── indexing/       faq_indexer, documents_indexer, file_readers
scripts/            run_indexers.py
devops/             Dockerfile, docker-compose.yml, .env.example
docs/               openapi.yaml
```