"""Клиент whole-file извлечения Qwen DashScope (OpenAI-compatible Files API)."""

from __future__ import annotations

import json
import logging
import os
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


def dashscope_api_key() -> str:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or ""
    ).strip()


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
    client = _openai_client(base_url)
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
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_id": file_id}},
                    {"type": "text", "text": prompt},
                ],
            },
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
        doc_model: str = "qwen-doc-turbo",
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

        file_id = upload_file(path, base_url=self.base_url)
        model = self._model_for_route(gate.route)
        raw = _call_extract_model(
            base_url=self.base_url,
            model=model,
            file_id=file_id,
            prompt=prompt,
            schema_hint=schema_hint,
        )
        parsed = _parse_model_result(raw, result_cls)
        self.cache.put(
            kind=kind,
            content_hash=digest,
            file_id=file_id,
            route=str(gate.route),
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
