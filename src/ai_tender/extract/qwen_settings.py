"""Общие настройки Qwen extract (тендер + каталог)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..models import Settings
from .qwen_extract import QwenExtractor, dashscope_api_key


def uses_qwen_extract(settings: Settings | None) -> bool:
    if settings is None:
        return False
    return (
        str(getattr(settings, "extract_backend", "legacy")).lower().strip() == "qwen"
        and bool(dashscope_api_key())
    )


def build_qwen_extractor(
    settings: Settings,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> QwenExtractor:
    return QwenExtractor(
        cache_dir=settings.cache_dir,
        base_url=settings.qwen_base_url,
        doc_model=settings.qwen_doc_model,
        long_model=settings.qwen_long_model,
        vl_model=settings.qwen_vl_model,
        vl_enabled=settings.vl_enabled,
        vl_pages_per_call=settings.qwen_vl_pages_per_call,
        vl_max_pages=settings.qwen_vl_max_pages,
        schema_version=settings.qwen_extract_schema_version,
        on_progress=on_progress,
    )
