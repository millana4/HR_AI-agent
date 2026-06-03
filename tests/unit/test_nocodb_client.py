"""Unit-тесты NocoDBClient с моком HTTP."""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.core.exceptions import RepositoryError
from app.repositories.nocodb_client import NocoDBClient


@pytest.fixture
def client():
    return NocoDBClient()


def _make_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    resp.text = text
    return resp


async def test_list_records_single_page(client: NocoDBClient):
    """Одна страница — возвращаем как есть."""
    fake_response = _make_response(
        200,
        {
            "list": [
                {"Id": 1, "FIO": "Иванов И.И."},
                {"Id": 2, "FIO": "Петров П.П."},
            ],
            "pageInfo": {"isLastPage": True},
        },
    )
    client._http.get = AsyncMock(return_value=fake_response)

    records = await client.list_records(table_id="tbl_abc")
    assert len(records) == 2
    assert records[0]["FIO"] == "Иванов И.И."


async def test_list_records_multiple_pages(client: NocoDBClient):
    """Если несколько страниц — собираем все."""
    page1 = _make_response(
        200,
        {
            "list": [{"Id": i} for i in range(1, 4)],
            "pageInfo": {"isLastPage": False},
        },
    )
    page2 = _make_response(
        200,
        {
            "list": [{"Id": i} for i in range(4, 6)],
            "pageInfo": {"isLastPage": True},
        },
    )
    client._http.get = AsyncMock(side_effect=[page1, page2])

    records = await client.list_records(table_id="tbl_abc", limit=3)
    assert len(records) == 5
    assert records[0]["Id"] == 1
    assert records[-1]["Id"] == 5


async def test_list_records_http_error(client: NocoDBClient):
    client._http.get = AsyncMock(
        return_value=_make_response(500, text="Internal Server Error")
    )

    with pytest.raises(RepositoryError):
        await client.list_records(table_id="tbl_abc")


async def test_list_records_invalid_json(client: NocoDBClient):
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json = MagicMock(side_effect=ValueError("invalid"))
    fake_response.text = "<html>"
    client._http.get = AsyncMock(return_value=fake_response)

    with pytest.raises(RepositoryError):
        await client.list_records(table_id="tbl_abc")


async def test_list_records_passes_fields(client: NocoDBClient):
    """Проверяем, что параметр fields передаётся в запрос."""
    fake_response = _make_response(
        200, {"list": [], "pageInfo": {"isLastPage": True}}
    )
    client._http.get = AsyncMock(return_value=fake_response)

    await client.list_records(table_id="tbl_abc", fields=["FIO", "Name"])

    call_args = client._http.get.call_args
    assert call_args.kwargs["params"]["fields"] == "FIO,Name"