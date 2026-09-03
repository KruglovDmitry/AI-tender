"""Тесты загрузки тендера по URL (VL-модуль)."""

from __future__ import annotations

from pathlib import Path

import httpx

from ai_tender.services.tender_fetch.browser import (
    PageCapture,
    filter_candidate_links,
    resolve_absolute_url,
)
from ai_tender.services.tender_fetch.downloader import filename_from_response
from ai_tender.services.tender_fetch.vl_page import match_links_to_candidates
from ai_tender.services.tender_fetch.schemas import TenderDocumentLink, TenderPageAnalysis


def test_filter_candidate_links_keeps_doc_extensions() -> None:
    links = [
        ("https://example.com/a.pdf", "Извещение"),
        ("https://example.com/news", "Новости"),
        ("javascript:void(0)", "skip"),
    ]
    out = filter_candidate_links(links)
    assert out == [("https://example.com/a.pdf", "Извещение")]


def test_filter_candidate_links_keeps_tender_keywords() -> None:
    links = [
        ("https://zakupki.gov.ru/epz/order/notice/ea20/view/documents.html?id=1", "Документы"),
    ]
    out = filter_candidate_links(links)
    assert len(out) == 1


def test_resolve_absolute_url() -> None:
    assert resolve_absolute_url(
        "https://zakupki.gov.ru/epz/order/notice/ea20/view/documents.html",
        "/file.pdf",
    ) == "https://zakupki.gov.ru/file.pdf"


def test_filename_from_content_disposition() -> None:
    headers = httpx.Headers({"content-disposition": 'attachment; filename="TZ.pdf"'})
    assert filename_from_response("https://x/y", headers, "fallback") == "TZ.pdf"


def test_match_links_to_candidates() -> None:
    capture = PageCapture(
        url="https://example.com/t",
        final_url="https://example.com/t",
        title="T",
        candidate_links=[
            ("https://example.com/doc1.pdf", "ТЗ"),
            ("https://example.com/doc2.zip", "Архив"),
        ],
    )
    analysis = TenderPageAnalysis(
        document_links=[
            TenderDocumentLink(url="https://example.com/doc1.pdf", title="Техническое задание"),
            TenderDocumentLink(url="https://evil.com/hack.pdf", title="bad"),
        ]
    )
    matched = match_links_to_candidates(analysis, capture)
    assert matched == [("https://example.com/doc1.pdf", "Техническое задание")]


def test_download_documents_saves_file(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        headers = httpx.Headers({"content-type": "application/pdf"})

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int = 65536):
            yield b"%PDF-1.4 test"

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def stream(self, method: str, url: str):
            return FakeResponse()

    monkeypatch.setattr(
        "ai_tender.services.tender_fetch.downloader.httpx.Client",
        FakeClient,
    )

    from ai_tender.services.tender_fetch.downloader import download_documents

    saved, warnings = download_documents(
        [("https://example.com/spec.pdf", "Спецификация")],
        tmp_path,
    )
    assert not warnings
    assert len(saved) == 1
    assert saved[0].read_bytes().startswith(b"%PDF")
