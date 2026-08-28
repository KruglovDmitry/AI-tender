"""Общие зависимости API."""

from __future__ import annotations

import os
from pathlib import Path

from ai_tender.models import get_settings


def is_running_in_docker() -> bool:
    return os.environ.get("RUNNING_IN_DOCKER") == "1" or Path("/.dockerenv").exists()


def default_tender_path() -> str:
    if is_running_in_docker():
        for candidate in (Path("/data/tender/1"), Path("/data/tender")):
            if candidate.is_dir():
                return str(candidate)
        return "/data/tender"
    return str((Path.cwd() / "sources" / "1").resolve())


def default_assets_path() -> str:
    if is_running_in_docker():
        return "/data/assets"
    return str((Path.cwd() / "assets").resolve())


def resolve_assets_path(path: str | None = None) -> Path:
    raw = path or default_assets_path()
    return Path(raw).expanduser().resolve()


def get_runtime_settings():
    return get_settings()
