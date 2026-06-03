"""
Smoke-тест: отправляет реальный запрос в GigaChat.

Запуск:
    python scripts/test_gigachat_real.py

Что проверяет:
    1. OAuth-авторизация работает
    2. Простой текстовый запрос проходит
    3. Запрос с tools возвращает function_call
"""
import asyncio

from app.core.logging import generate_correlation_id, get_logger, setup_logging
from app.llm.base import Message, ToolSpec
from app.llm.factory import get_llm_client


async def main():
    setup_logging()
    logger = get_logger(__name__)
    cid = generate_correlation_id()

    client = get_llm_client()
    try:
        # Тест 1: простой текстовый запрос
        print("\n=== Тест 1: простой текст ===")
        response = await client.chat(
            messages=[
                Message(role="system", content="Ты вежливый ассистент. Отвечай кратко."),
                Message(role="user", content="Привет! Скажи 'Тест прошёл' и больше ничего."),
            ],
            correlation_id=cid,
        )
        print(f"Тип ответа: {response.type}")
        print(f"Текст: {response.content}")

        # Тест 2: запрос с tools — LLM должна вызвать tool
        print("\n=== Тест 2: запрос с tools ===")
        tools = [
            ToolSpec(
                name="search_contacts",
                description="Поиск контактов сотрудника компании по имени и фамилии",
                parameters={
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string", "description": "Имя"},
                        "last_name": {"type": "string", "description": "Фамилия"},
                    },
                    "required": ["last_name"],
                },
            ),
        ]
        response = await client.chat(
            messages=[
                Message(role="system", content="Ты HR-ассистент. Используй tools для поиска."),
                Message(role="user", content="Найди телефон Иванова Ивана"),
            ],
            tools=tools,
            correlation_id=cid,
        )
        print(f"Тип ответа: {response.type}")
        if response.type == "tool_calls":
            for tc in response.tool_calls:
                print(f"Tool: {tc.name}, args: {tc.args}")
        else:
            print(f"Текст: {response.content}")
            print("⚠️  Ожидался tool_call, но пришёл текст")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())