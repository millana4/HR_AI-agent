"""Тесты репозитория документов."""
from unittest.mock import AsyncMock

import pytest

from app.repositories.documents import (
    Document,
    fetch_documents,
)


@pytest.fixture
def mock_client():
    return AsyncMock()


async def test_fetch_documents_filters_inactive(mock_client):
    """Записи с Active=false должны фильтроваться."""
    mock_client.list_records.return_value = [
        {
            "Id": 1,
            "Title": "Doc 1",
            "URL": "https://cdn.example.com/doc1.pdf",
            "Type": "support",
            "Description": "desc",
            "Active": True,
            "UpdatedAt": "2026-06-02 09:19:05+00:00",
        },
        {
            "Id": 2,
            "Title": "Doc 2",
            "URL": "https://cdn.example.com/doc2.pdf",
            "Type": "support",
            "Description": "desc",
            "Active": False,
            "UpdatedAt": "2026-06-02 09:19:05+00:00",
        },
    ]

    docs = await fetch_documents(mock_client)
    assert len(docs) == 1
    assert docs[0].id == 1


async def test_fetch_documents_skips_unknown_type(mock_client):
    """Записи с неизвестным Type пропускаются."""
    mock_client.list_records.return_value = [
        {
            "Id": 1,
            "Title": "Bad Doc",
            "URL": "https://cdn.example.com/x.pdf",
            "Type": "unknown_type",
            "Description": "",
            "Active": True,
            "UpdatedAt": "2026-06-02 09:19:05+00:00",
        },
    ]
    docs = await fetch_documents(mock_client)
    assert docs == []


async def test_fetch_documents_parses_all_three_types(mock_client):
    """blank, regulation, support — все валидные."""
    mock_client.list_records.return_value = [
        {
            "Id": 1, "Title": "B", "URL": "u1", "Type": "blank",
            "Description": "", "Active": True, "UpdatedAt": None,
        },
        {
            "Id": 2, "Title": "R", "URL": "u2", "Type": "regulation",
            "Description": "", "Active": True, "UpdatedAt": None,
        },
        {
            "Id": 3, "Title": "S", "URL": "u3", "Type": "support",
            "Description": "", "Active": True, "UpdatedAt": None,
        },
    ]
    docs = await fetch_documents(mock_client)
    assert len(docs) == 3
    assert {d.doc_type for d in docs} == {"blank", "regulation", "support"}


async def test_fetch_documents_parses_dates(mock_client):
    """ISO-строка превращается в datetime."""
    mock_client.list_records.return_value = [
        {
            "Id": 1, "Title": "T", "URL": "u", "Type": "support",
            "Description": "", "Active": True,
            "UpdatedAt": "2026-06-02 09:19:05+00:00",
        },
    ]
    docs = await fetch_documents(mock_client)
    assert docs[0].updated_at is not None
    assert docs[0].updated_at.year == 2026


async def test_fetch_documents_handles_missing_dates(mock_client):
    """Если UpdatedAt=null — updated_at=None в dataclass."""
    mock_client.list_records.return_value = [
        {
            "Id": 1, "Title": "T", "URL": "u", "Type": "support",
            "Description": "", "Active": True,
            "UpdatedAt": None,
        },
    ]
    docs = await fetch_documents(mock_client)
    assert docs[0].updated_at is None