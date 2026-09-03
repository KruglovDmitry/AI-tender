"""Загрузка документов тендера по URL через VL-анализ страницы."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ...models import Settings
from ..upload_service import clear_directory, expand_top_level_archives
from .browser import capture_tender_page
from .downloader import download_documents
from .vl_page import analyze_page_with_vl, match_links_to_candidates

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


def fetch_tender_from_url(
    url: str,
    dest: Path,
    *,
    settings: Settings,
    progress: ProgressCallback | None = None,
    max_screenshots: int = 4,
) -> tuple[Path, list[str]]:
    """Открывает страницу тендера, VL находит документы, скачивает в dest."""

    def emit(message: str, fraction: float) -> None:
        if progress:
            progress(message, fraction)
        logger.info("[tender_fetch] %s (%.0f%%)", message, fraction * 100)

    dest = dest.expanduser().resolve()
    clear_directory(dest)
    warnings: list[str] = []

    emit("Открываем страницу тендера…", 0.05)
    capture = capture_tender_page(
        url,
        max_screenshots=max_screenshots,
    )
    if not capture.candidate_links:
        warnings.append(
            "На странице не найдено явных ссылок на документы — VL попробует по скриншотам."
        )

    emit(
        f"VL анализирует страницу ({len(capture.screenshots)} скр.)…",
        0.25,
    )
    analysis = analyze_page_with_vl(capture, settings=settings)
    if analysis.page_summary:
        logger.info("VL summary: %s", analysis.page_summary[:300])

    links = match_links_to_candidates(analysis, capture)
    if not links and analysis.document_links:
        warnings.append(
            "VL нашёл документы, но URL не совпали со ссылками страницы — скачивание пропущено."
        )
    if not links:
        raise ValueError(
            "VL не нашёл документов для скачивания на странице. "
            f"Заголовок: {capture.title or url}"
        )

    emit(f"Скачиваем {len(links)} документ(ов)…", 0.45)
    saved, dl_warnings = download_documents(
        links,
        dest,
        referer=capture.final_url,
    )
    warnings.extend(dl_warnings)
    if not saved:
        raise ValueError("Не удалось скачать ни одного документа тендера.")

    emit("Распаковка архивов…", 0.85)
    expand_top_level_archives(dest, warnings)

    if not any(path.is_file() for path in dest.rglob("*")):
        raise ValueError("После скачивания не осталось файлов (пустые архивы?).")

    meta_bits = []
    if analysis.tender_title:
        meta_bits.append(f"тендер: {analysis.tender_title}")
    if analysis.notice_number:
        meta_bits.append(f"№ {analysis.notice_number}")
    if meta_bits:
        warnings.insert(0, "VL: " + "; ".join(meta_bits))

    emit(f"Готово: {len(saved)} файл(ов)", 0.95)
    return dest, warnings
