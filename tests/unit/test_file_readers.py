"""Тесты ридеров файлов."""
import io

import httpx
import pytest

from app.core.exceptions import RepositoryError
from app.indexing.file_readers import (
    _detect_format,
    _extract_docx,
    _extract_pdf,
    _extract_txt,
    download_and_extract,
)


# --- _detect_format ---

def test_detect_format_pdf():
    assert _detect_format("https://cdn.example.com/doc.pdf") == "pdf"
    assert _detect_format("https://cdn.example.com/DOC.PDF") == "pdf"


def test_detect_format_docx():
    assert _detect_format("https://cdn.example.com/doc.docx") == "docx"
    assert _detect_format("https://cdn.example.com/doc.DOCX") == "docx"


def test_detect_format_doc_legacy():
    """.doc (старый формат) тоже отдаём в docx-ридер."""
    assert _detect_format("https://cdn.example.com/doc.doc") == "docx"


def test_detect_format_txt():
    assert _detect_format("https://cdn.example.com/file.txt") == "txt"


def test_detect_format_unknown_raises():
    with pytest.raises(RepositoryError, match="Cannot detect format"):
        _detect_format("https://cdn.example.com/file.xlsx")


def test_detect_format_no_extension_raises():
    with pytest.raises(RepositoryError, match="Cannot detect format"):
        _detect_format("https://cdn.example.com/file")


# --- _extract_pdf ---

def _make_minimal_pdf(text: str) -> bytes:
    """Создать минимальный валидный PDF с текстом для теста."""
    from pypdf import PdfWriter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, text)
    c.save()
    return buf.getvalue()


def test_extract_pdf_with_text():
    """PDF с текстовым слоем извлекается."""
    pytest.importorskip("reportlab")
    content = _make_minimal_pdf("Test content for PDF")
    text = _extract_pdf(content, url="test.pdf", correlation_id="-")
    assert "Test content" in text


def test_extract_pdf_invalid_bytes_raises():
    with pytest.raises(RepositoryError, match="Failed to parse PDF"):
        _extract_pdf(b"not a pdf", url="bad.pdf", correlation_id="-")


# --- _extract_docx ---

def _make_minimal_docx(paragraphs: list[str]) -> bytes:
    """Создать DOCX с заданными параграфами."""
    from docx import Document as DocxDocument
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_docx_with_paragraphs():
    content = _make_minimal_docx(["Первый параграф", "Второй параграф"])
    text = _extract_docx(content, url="test.docx", correlation_id="-")
    assert "Первый параграф" in text
    assert "Второй параграф" in text


def test_extract_docx_invalid_bytes_raises():
    with pytest.raises(RepositoryError, match="Failed to parse DOCX"):
        _extract_docx(b"not a docx", url="bad.docx", correlation_id="-")


# --- _extract_txt ---

def test_extract_txt_utf8():
    content = "Привет, мир".encode("utf-8")
    text = _extract_txt(content, url="x.txt", correlation_id="-")
    assert text == "Привет, мир"


def test_extract_txt_cp1251_fallback():
    """Если не UTF-8, пробуем cp1251."""
    content = "Привет".encode("cp1251")
    text = _extract_txt(content, url="x.txt", correlation_id="-")
    assert text == "Привет"


# --- download_and_extract (integration with mock httpx) ---

async def test_download_and_extract_txt(monkeypatch):
    """Скачивание + извлечение TXT через мок httpx."""
    payload = "Hello world".encode("utf-8")

    class FakeResponse:
        content = payload
        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    text = await download_and_extract("https://cdn.example.com/x.txt")
    assert text == "Hello world"


async def test_download_and_extract_http_error_raises(monkeypatch):
    """HTTP 404 при скачивании → RepositoryError."""
    class FakeResponse:
        status_code = 404
        text = "Not Found"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url):
            raise httpx.HTTPStatusError(
                "404", request=None, response=FakeResponse(),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(RepositoryError, match="Failed to download"):
        await download_and_extract("https://cdn.example.com/missing.pdf")