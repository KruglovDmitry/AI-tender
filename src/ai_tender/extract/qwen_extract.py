"""Клиент whole-file извлечения Qwen DashScope (OpenAI-compatible Files API)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from .page_images import render_document_images
from .qwen_cache import QwenExtractCache, content_sha256
from .qwen_gate import ExtractRoute, GateDecision, can_send_to_qwen
from .schemas import (
    EXTRACT_SCHEMA_VERSION,
    CatalogExtractResult,
    ProductRecord,
    TenderExtractResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DATA_EGRESS_WARNING = (
    "Внимание: файл будет загружен на серверы DashScope (Qwen) для извлечения."
)
# qwen-long / qwen-doc-turbo читают fileid://; chat-модели intl (qwen-plus) — только локальный текст.
FILE_EXTRACT_NATIVE_MODELS = frozenset({"qwen-long", "qwen-doc-turbo"})
MAX_LOCAL_TEXT_CHARS = 120_000


def dashscope_api_key() -> str:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or ""
    ).strip()


def uses_native_file_extract(model: str) -> bool:
    return model.strip().casefold() in FILE_EXTRACT_NATIVE_MODELS


def _read_local_document_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import fitz

        with fitz.open(path) as doc:
            parts = [(doc.load_page(i).get_text() or "") for i in range(doc.page_count)]
        return "\n\n".join(parts).strip()
    if ext in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if ext == ".docx":
        import docx2txt

        return (docx2txt.process(str(path)) or "").strip()
    if ext == ".doc":
        from ..services.loader_service import extract_doc_text

        text, error = extract_doc_text(path)
        if not text.strip():
            raise ValueError(error or "пустой текст .doc")
        return text.strip()
    raise ValueError(f"локальный текст не поддержан для {ext}")


def _openai_client(base_url: str):
    from openai import OpenAI

    return OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=dashscope_api_key() or "MISSING",
        timeout=300.0,
    )


def upload_file(path: Path, *, base_url: str) -> str:
    """upload → file_id (purpose=file-extract)."""
    client = _openai_client(base_url)
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="file-extract")
    file_id = str(getattr(uploaded, "id", "") or "")
    if not file_id:
        raise RuntimeError("DashScope files.create не вернул file_id")
    logger.info("Qwen upload %s → file_id=%s", path.name, file_id)
    return file_id


def _call_extract_model(
    *,
    base_url: str,
    model: str,
    file_id: str,
    prompt: str,
    schema_hint: str,
) -> str:
    from openai import BadRequestError

    client = _openai_client(base_url)
    messages = [
        {
            "role": "system",
            "content": (
                "Верни ТОЛЬКО валидный JSON без markdown. "
                f"Структура: {schema_hint}"
            ),
        },
        {"role": "system", "content": f"fileid://{file_id}"},
        {"role": "user", "content": prompt},
    ]
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            choice = response.choices[0] if response.choices else None
            return (choice.message.content or "") if choice else ""
        except BadRequestError as exc:
            err = str(exc)
            if "File parsing in progress" in err and attempt + 1 < max_retries:
                logger.info(
                    "Qwen file parsing in progress (%s/%s), retry…",
                    attempt + 1,
                    max_retries,
                )
                time.sleep(2)
                continue
            raise
    return ""


def _call_text_extract_model(
    *,
    base_url: str,
    model: str,
    document_text: str,
    prompt: str,
    schema_hint: str,
) -> str:
    client = _openai_client(base_url)
    body = document_text[:MAX_LOCAL_TEXT_CHARS]
    if len(document_text) > MAX_LOCAL_TEXT_CHARS:
        logger.warning(
            "Qwen local-text extract truncated %s → %s chars",
            len(document_text),
            MAX_LOCAL_TEXT_CHARS,
        )
    user_content = f"{prompt}\n\n--- ДОКУМЕНТ ---\n{body}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Верни ТОЛЬКО валидный JSON без markdown. "
                    f"Структура: {schema_hint}"
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    choice = response.choices[0] if response.choices else None
    return (choice.message.content or "") if choice else ""


def _parse_model_result(raw: str, model_cls: type[T]) -> T:
    from ..providers import try_parse_llm_json

    data = try_parse_llm_json(raw)
    if data is None:
        raise ValueError(f"Qwen вернул невалидный JSON: {raw[:500]!r}")
    return model_cls.model_validate(data)


def _call_vl_extract_model(
    *,
    base_url: str,
    model: str,
    prompt: str,
    schema_hint: str,
    image_data_urls: list[str],
    page_note: str,
) -> str:
    client = _openai_client(base_url)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{prompt}\n\n{page_note}\n"
                "Верни ТОЛЬКО валидный JSON без markdown."
            ),
        }
    ]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Верни ТОЛЬКО валидный JSON без markdown. "
                    f"Структура: {schema_hint}"
                ),
            },
            {"role": "user", "content": content},
        ],
        temperature=0,
        max_tokens=4096,
    )
    choice = response.choices[0] if response.choices else None
    return (choice.message.content or "") if choice else ""


def _product_key(product: ProductRecord) -> str:
    model = (product.model or "").strip().casefold()
    if model:
        return f"m:{model}"
    desc = (product.canonical_desc or product.raw_chunk or "").strip().casefold()
    return f"d:{desc[:120]}"


def _merge_catalog_results(parts: list[CatalogExtractResult]) -> CatalogExtractResult:
    catalog_name = next((p.catalog_name.strip() for p in parts if p.catalog_name.strip()), "")
    seen: set[str] = set()
    products: list[ProductRecord] = []
    for part in parts:
        for product in part.products:
            key = _product_key(product)
            if not key or key in {"m:", "d:"}:
                continue
            if key in seen:
                continue
            seen.add(key)
            products.append(product)
    return CatalogExtractResult(catalog_name=catalog_name, products=products)


def _merge_tender_results(parts: list[TenderExtractResult]) -> TenderExtractResult:
    if not parts:
        return TenderExtractResult()
    from .schemas import ScopeItemExtract

    by_name: dict[str, ScopeItemExtract] = {}
    summaries: list[str] = []
    missing: list[str] = []
    needs_more = False
    confidences: list[float] = []
    for part in parts:
        if part.scope_summary.strip():
            summaries.append(part.scope_summary.strip())
        if part.missing_signals.strip():
            missing.append(part.missing_signals.strip())
        needs_more = needs_more or part.needs_more_docs
        confidences.append(float(part.overall_confidence or 0.0))
        for item in part.scope_items:
            key = (item.name or "").strip().casefold()
            if not key:
                continue
            prev = by_name.get(key)
            if prev is None or len(item.requirements) > len(prev.requirements):
                by_name[key] = item
    return TenderExtractResult(
        scope_summary=" ".join(summaries).strip(),
        scope_items=list(by_name.values()),
        overall_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        needs_more_docs=needs_more,
        missing_signals="; ".join(missing),
    )


class QwenExtractor:
    """Whole-file extract с gate, кэшем и разделением doc vs scan/VL."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        doc_model: str = "qwen-plus",
        long_model: str = "qwen-long",
        vl_model: str = "qwen-vl-plus",
        vl_enabled: bool = True,
        vl_pages_per_call: int = 2,
        vl_max_pages: int = 80,
        schema_version: str = EXTRACT_SCHEMA_VERSION,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.cache = QwenExtractCache(cache_dir)
        self.base_url = base_url
        self.doc_model = doc_model
        self.long_model = long_model
        self.vl_model = vl_model
        self.vl_enabled = vl_enabled
        self.vl_pages_per_call = max(1, int(vl_pages_per_call))
        self.vl_max_pages = max(1, int(vl_max_pages))
        self.schema_version = schema_version
        self.on_progress = on_progress

    def gate(self, path: Path, *, purpose: str) -> GateDecision:
        return can_send_to_qwen(path, purpose=purpose)

    def should_use_vl(self, gate: GateDecision) -> bool:
        """VL: нет текстового слоя/картинка, либо принудительно в настройках."""
        return self.vl_enabled or gate.route == ExtractRoute.qwen_scan

    def _model_for_route(self, route: ExtractRoute) -> str:
        if route == ExtractRoute.qwen_long:
            return self.long_model
        return self.doc_model

    def _emit(self, message: str) -> None:
        logger.info("%s", message)
        if self.on_progress:
            self.on_progress(message)

    def extract_json(
        self,
        path: Path,
        *,
        kind: str,
        gate: GateDecision,
        prompt: str,
        schema_hint: str,
        result_cls: type[T],
    ) -> T:
        if not gate.sends_to_qwen_doc:
            raise ValueError(f"extract_json вызван для маршрута {gate.route}: {gate.reason}")

        logger.warning(DATA_EGRESS_WARNING + " file=%s", path.name)
        digest = content_sha256(path)
        cached = self.cache.get(kind=kind, content_hash=digest, schema_version=self.schema_version)
        if cached and cached.get("result"):
            logger.info("Qwen cache hit %s (%s)", path.name, digest[:12])
            return result_cls.model_validate(cached["result"])

        model = self._model_for_route(gate.route)
        if uses_native_file_extract(model):
            file_id = upload_file(path, base_url=self.base_url)
            raw = _call_extract_model(
                base_url=self.base_url,
                model=model,
                file_id=file_id,
                prompt=prompt,
                schema_hint=schema_hint,
            )
            route = str(gate.route)
        else:
            local_text = _read_local_document_text(path)
            if not local_text.strip():
                raise ValueError(f"не удалось извлечь локальный текст из {path.name}")
            file_id = ""
            route = "qwen_local_text"
            logger.info(
                "Qwen local-text extract %s via %s (%s chars)",
                path.name,
                model,
                len(local_text),
            )
            raw = _call_text_extract_model(
                base_url=self.base_url,
                model=model,
                document_text=local_text,
                prompt=prompt,
                schema_hint=schema_hint,
            )
        parsed = _parse_model_result(raw, result_cls)
        self.cache.put(
            kind=kind,
            content_hash=digest,
            file_id=file_id,
            route=route,
            model=model,
            result=json.loads(parsed.model_dump_json()),
            schema_version=self.schema_version,
        )
        return parsed

    def extract_vl_json(
        self,
        path: Path,
        *,
        kind: str,
        prompt: str,
        schema_hint: str,
        result_cls: type[T],
        merge_parts: Callable[[list[T]], T],
    ) -> T:
        if not dashscope_api_key():
            raise ValueError("Для VL задайте QWEN_API_KEY или DASHSCOPE_API_KEY")

        logger.warning(DATA_EGRESS_WARNING + " VL file=%s", path.name)
        digest = content_sha256(path)
        cache_kind = f"{kind}_vl"
        cached = self.cache.get(
            kind=cache_kind,
            content_hash=digest,
            schema_version=self.schema_version,
        )
        if cached and cached.get("result"):
            logger.info("Qwen VL cache hit %s (%s)", path.name, digest[:12])
            return result_cls.model_validate(cached["result"])

        pages = render_document_images(path, max_pages=self.vl_max_pages)
        self._emit(f"VL {path.name}: {len(pages)} стр., модель {self.vl_model}")
        parts: list[T] = []
        step = self.vl_pages_per_call
        for start in range(0, len(pages), step):
            batch = pages[start : start + step]
            page_nos = [str(p[0]) for p in batch]
            page_note = f"Страницы документа: {', '.join(page_nos)} (всего {len(pages)})."
            self._emit(f"VL {path.name}: стр. {', '.join(page_nos)}/{len(pages)}")
            raw = _call_vl_extract_model(
                base_url=self.base_url,
                model=self.vl_model,
                prompt=prompt,
                schema_hint=schema_hint,
                image_data_urls=[p[1] for p in batch],
                page_note=page_note,
            )
            parts.append(_parse_model_result(raw, result_cls))

        parsed = merge_parts(parts)
        self.cache.put(
            kind=cache_kind,
            content_hash=digest,
            file_id="",
            route="qwen_scan",
            model=self.vl_model,
            result=json.loads(parsed.model_dump_json()),
            schema_version=self.schema_version,
        )
        return parsed

    def extract_catalog(self, path: Path, *, prompt: str) -> CatalogExtractResult:
        gate = self.gate(path, purpose="catalog")
        hint = '{"catalog_name":"","products":[{"model":"","characteristics":[]}]}'
        if self.should_use_vl(gate):
            return self.extract_vl_json(
                path,
                kind="catalog",
                prompt=prompt,
                schema_hint=hint,
                result_cls=CatalogExtractResult,
                merge_parts=_merge_catalog_results,
            )
        if not gate.sends_to_qwen_doc:
            raise ValueError(gate.reason)
        return self.extract_json(
            path,
            kind="catalog",
            gate=gate,
            prompt=prompt,
            schema_hint=hint,
            result_cls=CatalogExtractResult,
        )

    def extract_tender(self, path: Path, *, prompt: str) -> TenderExtractResult:
        gate = self.gate(path, purpose="tender")
        hint = (
            '{"scope_summary":"","scope_items":[{"name":"","qty":1,"unit":"шт.",'
            '"requirements":[{"text":"","kind":"specs"}]}]}'
        )
        if self.should_use_vl(gate):
            return self.extract_vl_json(
                path,
                kind="tender",
                prompt=prompt,
                schema_hint=hint,
                result_cls=TenderExtractResult,
                merge_parts=_merge_tender_results,
            )
        if not gate.sends_to_qwen_doc:
            raise ValueError(gate.reason)
        return self.extract_json(
            path,
            kind="tender",
            gate=gate,
            prompt=prompt,
            schema_hint=hint,
            result_cls=TenderExtractResult,
        )
