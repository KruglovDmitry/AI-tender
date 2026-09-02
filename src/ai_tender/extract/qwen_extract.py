"""Клиент whole-file извлечения Qwen DashScope (OpenAI-compatible Files API)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .qwen_cache import QwenExtractCache, content_sha256
from .qwen_gate import ExtractRoute, GateDecision, can_send_to_qwen
from .schemas import (
    EXTRACT_SCHEMA_VERSION,
    CatalogExtractResult,
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


class QwenExtractor:
    """Whole-file extract с gate, кэшем и разделением doc vs scan."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        doc_model: str = "qwen-plus",
        long_model: str = "qwen-long",
        schema_version: str = EXTRACT_SCHEMA_VERSION,
    ) -> None:
        self.cache = QwenExtractCache(cache_dir)
        self.base_url = base_url
        self.doc_model = doc_model
        self.long_model = long_model
        self.schema_version = schema_version

    def gate(self, path: Path, *, purpose: str) -> GateDecision:
        return can_send_to_qwen(path, purpose=purpose)

    def _model_for_route(self, route: ExtractRoute) -> str:
        if route == ExtractRoute.qwen_long:
            return self.long_model
        return self.doc_model

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

    def extract_catalog(self, path: Path, *, prompt: str) -> CatalogExtractResult:
        gate = self.gate(path, purpose="catalog")
        if gate.route == ExtractRoute.qwen_scan:
            raise NotImplementedError(
                "Скан каталога — отдельный scan-контракт (qwen-vl), не doc-extract"
            )
        if not gate.sends_to_qwen_doc:
            raise ValueError(gate.reason)
        hint = '{"catalog_name":"","products":[{"model":"","characteristics":[]}]}'
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
        if gate.route == ExtractRoute.qwen_scan:
            raise NotImplementedError(
                "Скан тендера — отдельный scan-контракт (qwen-vl), не doc-extract"
            )
        if not gate.sends_to_qwen_doc:
            raise ValueError(gate.reason)
        hint = (
            '{"scope_summary":"","scope_items":[{"name":"","qty":1,"unit":"шт.",'
            '"requirements":[{"text":"","kind":"specs"}]}]}'
        )
        return self.extract_json(
            path,
            kind="tender",
            gate=gate,
            prompt=prompt,
            schema_hint=hint,
            result_cls=TenderExtractResult,
        )
