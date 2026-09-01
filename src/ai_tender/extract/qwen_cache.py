"""Кэш DashScope file_id и результата извлечения по хешу контента + версии схемы."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .schemas import EXTRACT_SCHEMA_VERSION

ExtractKind = Literal["catalog", "tender"]


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QwenExtractCache:
    def __init__(self, cache_dir: Path) -> None:
        self.root = cache_dir.expanduser().resolve() / "qwen_extract"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(
        self,
        *,
        kind: ExtractKind,
        content_hash: str,
        schema_version: str,
    ) -> Path:
        safe = content_hash[:32]
        return self.root / kind / f"{safe}_v{schema_version}.json"

    def get(
        self,
        *,
        kind: ExtractKind,
        content_hash: str,
        schema_version: str = EXTRACT_SCHEMA_VERSION,
    ) -> dict[str, Any] | None:
        path = self._path(kind=kind, content_hash=content_hash, schema_version=schema_version)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("content_sha256") != content_hash:
            return None
        if str(data.get("schema_version")) != schema_version:
            return None
        return data

    def put(
        self,
        *,
        kind: ExtractKind,
        content_hash: str,
        file_id: str,
        route: str,
        result: dict[str, Any],
        schema_version: str = EXTRACT_SCHEMA_VERSION,
        model: str = "",
    ) -> Path:
        path = self._path(kind=kind, content_hash=content_hash, schema_version=schema_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "content_sha256": content_hash,
            "schema_version": schema_version,
            "file_id": file_id,
            "route": route,
            "model": model,
            "kind": kind,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
