"""Тест репозитория сотрудников."""
from unittest.mock import AsyncMock

import pytest

from app.repositories.pivot import fetch_pivot
from app.repositories.nocodb_client import NocoDBClient


async def test_fetch_pivot():
    client = NocoDBClient()
    client.list_records = AsyncMock(
        return_value=[
            {"Id": 1, "FIO": "Иванов Иван Иванович"},
            {"Id": 2, "FIO": "Петров Пётр"},
            {"Id": 3, "FIO": None},  # без ФИО — пропускаем
            {"Id": 4},  # совсем нет поля
            {"Id": 5, "FIO": ""},  # пустая строка — пропускаем
        ]
    )

    result = await fetch_pivot(client)
    assert result == ["Иванов Иван Иванович", "Петров Пётр"]