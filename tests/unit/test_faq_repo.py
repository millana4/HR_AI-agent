"""Тесты репозитория FAQ."""
from unittest.mock import AsyncMock

import pytest

from app.repositories.faq import fetch_faq


@pytest.fixture
def mock_client():
    return AsyncMock()


async def test_fetch_faq_filters_inactive(mock_client):
    mock_client.list_records.return_value = [
        {
            "Id": 1, "Question": "Q1", "Answer": "A1",
            "Link": None, "Attachment": None, "Hidden_data": None,
            "Active": True,
        },
        {
            "Id": 2, "Question": "Q2", "Answer": "A2",
            "Link": None, "Attachment": None, "Hidden_data": None,
            "Active": False,
        },
    ]
    entries = await fetch_faq(mock_client)
    assert len(entries) == 1
    assert entries[0].id == 1


async def test_fetch_faq_skips_empty_questions(mock_client):
    mock_client.list_records.return_value = [
        {
            "Id": 1, "Question": "", "Answer": "A",
            "Link": None, "Attachment": None, "Hidden_data": None,
            "Active": True,
        },
        {
            "Id": 2, "Question": "Q", "Answer": "",
            "Link": None, "Attachment": None, "Hidden_data": None,
            "Active": True,
        },
    ]
    entries = await fetch_faq(mock_client)
    assert entries == []


async def test_fetch_faq_parses_all_fields(mock_client):
    mock_client.list_records.return_value = [
        {
            "Id": 1,
            "Question": "Как пройти?",
            "Answer": "Через холл",
            "Link": "https://example.com",
            "Attachment": "https://cdn.example.com/x.pdf",
            "Hidden_data": "АД_ДИР=Иван Иванов",
            "Active": True,
        },
    ]
    entries = await fetch_faq(mock_client)
    assert len(entries) == 1
    e = entries[0]
    assert e.question == "Как пройти?"
    assert e.answer == "Через холл"
    assert e.link == "https://example.com"
    assert e.attachment == "https://cdn.example.com/x.pdf"
    assert e.hidden_data == "АД_ДИР=Иван Иванов"


async def test_fetch_faq_handles_null_optional_fields(mock_client):
    mock_client.list_records.return_value = [
        {
            "Id": 1, "Question": "Q", "Answer": "A",
            "Link": None, "Attachment": None, "Hidden_data": None,
            "Active": True,
        },
    ]
    entries = await fetch_faq(mock_client)
    assert entries[0].link is None
    assert entries[0].attachment is None
    assert entries[0].hidden_data is None