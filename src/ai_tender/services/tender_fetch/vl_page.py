"""VL-анализ скриншотов страницы тендера через Qwen-VL."""

from __future__ import annotations

import json
import logging
from typing import Any

from ...extract.qwen_extract import dashscope_api_key
from ...models import Settings
from ...providers import try_parse_llm_json
from .browser import PageCapture
from .schemas import (
    TENDER_PAGE_VL_PROMPT,
    TENDER_PAGE_VL_SCHEMA,
    TenderPageAnalysis,
)

logger = logging.getLogger(__name__)


def _openai_client(base_url: str):
    from openai import OpenAI

    return OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=dashscope_api_key() or "MISSING",
        timeout=300.0,
    )


def _format_candidate_links(links: list[tuple[str, str]], *, limit: int = 80) -> str:
    rows: list[dict[str, str]] = []
    for href, text in links[:limit]:
        rows.append({"href": href, "text": text or href})
    if len(links) > limit:
        rows.append({"href": "...", "text": f"… ещё {len(links) - limit} ссылок"})
    return json.dumps(rows, ensure_ascii=False, indent=2)


def analyze_page_with_vl(
    capture: PageCapture,
    *,
    settings: Settings,
) -> TenderPageAnalysis:
    if not dashscope_api_key():
        raise RuntimeError("Не задан QWEN_API_KEY или DASHSCOPE_API_KEY для VL-анализа страницы")

    if not capture.screenshots:
        raise ValueError("Нет скриншотов страницы для VL-анализа")

    client = _openai_client(settings.qwen_base_url)
    links_json = _format_candidate_links(capture.candidate_links)
    page_note = (
        f"URL страницы: {capture.final_url}\n"
        f"Заголовок вкладки: {capture.title}\n"
        f"Скриншотов: {len(capture.screenshots)}\n\n"
        f"candidate_links:\n{links_json}"
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{TENDER_PAGE_VL_PROMPT}\n\n{page_note}\n"
                "Верни ТОЛЬКО валидный JSON без markdown."
            ),
        }
    ]
    for data_url in capture.screenshots:
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    response = client.chat.completions.create(
        model=settings.qwen_vl_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Верни ТОЛЬКО валидный JSON без markdown. "
                    f"Структура: {TENDER_PAGE_VL_SCHEMA}"
                ),
            },
            {"role": "user", "content": content},
        ],
        temperature=0,
        max_tokens=4096,
    )
    choice = response.choices[0] if response.choices else None
    raw = (choice.message.content or "") if choice else ""
    data = try_parse_llm_json(raw)
    if data is None:
        raise ValueError(f"VL вернул невалидный JSON: {raw[:500]!r}")
    analysis = TenderPageAnalysis.model_validate(data)
    logger.info(
        "VL page analysis: %d links, confidence=%.2f, title=%r",
        len(analysis.document_links),
        analysis.confidence,
        analysis.tender_title[:80] if analysis.tender_title else "",
    )
    return analysis


def match_links_to_candidates(
    analysis: TenderPageAnalysis,
    capture: PageCapture,
) -> list[tuple[str, str]]:
    """Оставляет только URL, которые есть среди candidate_links (или совпадают по пути)."""
    allowed = {href for href, _ in capture.candidate_links}
    allowed_paths = {href.split("?", 1)[0] for href in allowed}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in analysis.document_links:
        url = item.url.strip()
        if not url or url in seen:
            continue
        base = url.split("?", 1)[0]
        if url in allowed or base in allowed_paths:
            seen.add(url)
            out.append((url, item.title or item.reason or url))
            continue
        for href, text in capture.candidate_links:
            if href == url or href.split("?", 1)[0] == base:
                seen.add(href)
                out.append((href, item.title or text or href))
                break
    return out
