"""Скриншоты страницы тендера через Playwright."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

DOC_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".zip",
    ".rar",
    ".7z",
    ".rtf",
    ".odt",
    ".ods",
)
TENDER_KEYWORDS = (
    "документ",
    "извещ",
    "техн",
    "задан",
    "контракт",
    "специф",
    "описан",
    "проект",
    "скач",
    "файл",
    "прилож",
    "download",
    "document",
)


@dataclass
class PageCapture:
    url: str
    final_url: str
    title: str
    screenshots: list[str] = field(default_factory=list)
    candidate_links: list[tuple[str, str]] = field(default_factory=list)


def _jpeg_data_url(png_bytes: bytes, *, quality: int = 82) -> str:
    from PIL import Image

    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _looks_like_document_link(href: str, text: str) -> bool:
    lower_href = href.casefold()
    lower_text = text.casefold()
    if any(ext in lower_href for ext in DOC_EXTENSIONS):
        return True
    if "download" in lower_href or "filestore" in lower_href or "document" in lower_href:
        return True
    return any(word in lower_text or word in lower_href for word in TENDER_KEYWORDS)


def filter_candidate_links(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for href, text in links:
        href = href.strip()
        if not href or href.startswith(("javascript:", "mailto:", "#")):
            continue
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"}:
            continue
        if href in seen:
            continue
        if not _looks_like_document_link(href, text):
            continue
        seen.add(href)
        out.append((href, text))
    return out


def capture_tender_page(
    url: str,
    *,
    max_screenshots: int = 4,
    viewport_width: int = 1280,
    viewport_height: int = 900,
    timeout_ms: int = 90_000,
) -> PageCapture:
    """Открывает URL, собирает ссылки и делает серию скриншотов при прокрутке."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Для загрузки по ссылке нужен playwright. "
            "Установите: pip install playwright && playwright install chromium"
        ) from exc

    normalized = url.strip()
    if not normalized:
        raise ValueError("Пустой URL тендера")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            locale="ru-RU",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        try:
            page.goto(normalized, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            raw_links: list[dict[str, str]] = page.evaluate(
                """() => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    href: a.href,
                    text: (a.innerText || a.getAttribute('title') || '').trim().slice(0, 240)
                }))"""
            )
            links = [(item["href"], item["text"]) for item in raw_links if item.get("href")]
            candidate_links = filter_candidate_links(links)

            screenshots: list[str] = []
            scroll_height = page.evaluate("() => document.body.scrollHeight")
            step = max(viewport_height - 120, 400)
            offsets = list(range(0, max(scroll_height, 1), step))
            if not offsets:
                offsets = [0]
            for offset in offsets[: max(1, max_screenshots)]:
                page.evaluate("(y) => window.scrollTo(0, y)", offset)
                page.wait_for_timeout(400)
                png = page.screenshot(full_page=False, type="png")
                screenshots.append(_jpeg_data_url(png))
                if len(screenshots) >= max_screenshots:
                    break

            return PageCapture(
                url=normalized,
                final_url=page.url,
                title=page.title(),
                screenshots=screenshots,
                candidate_links=candidate_links,
            )
        finally:
            context.close()
            browser.close()


def resolve_absolute_url(base_url: str, href: str) -> str:
    return urljoin(base_url, href.strip())
