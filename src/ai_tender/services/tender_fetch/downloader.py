"""Скачивание документов тендера по ссылкам."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from ..upload_service import safe_filename

logger = logging.getLogger(__name__)

DOWNLOADABLE_SUFFIXES = {
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
    ".txt",
}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.I)


def filename_from_response(url: str, headers: httpx.Headers, fallback_title: str) -> str:
    cd = headers.get("content-disposition") or ""
    match = _FILENAME_RE.search(cd)
    if match:
        name = unquote(match.group(1).strip())
        if name:
            return safe_filename(name)
    path_name = Path(unquote(urlparse(url).path)).name
    if path_name and "." in path_name:
        return safe_filename(path_name)
    title = fallback_title.strip() or "document"
    title = re.sub(r"[^\w\s\-.()]", "_", title, flags=re.UNICODE)
    title = title.strip("._ ") or "document"
    suffix = _guess_suffix(headers.get("content-type", ""))
    return safe_filename(f"{title}{suffix}")


def _guess_suffix(content_type: str) -> str:
    lower = content_type.casefold()
    mapping = {
        "pdf": ".pdf",
        "zip": ".zip",
        "msword": ".doc",
        "wordprocessingml": ".docx",
        "spreadsheetml": ".xlsx",
        "vnd.ms-excel": ".xls",
        "x-rar": ".rar",
    }
    for key, suffix in mapping.items():
        if key in lower:
            return suffix
    return ".bin"


def _unique_path(folder: Path, name: str) -> Path:
    target = folder / name
    if not target.exists():
        return target
    stem = Path(name).stem
    suffix = Path(name).suffix
    for idx in range(2, 100):
        candidate = folder / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    return folder / f"{stem}_{hash(name) & 0xFFFF}{suffix}"


def download_documents(
    links: list[tuple[str, str]],
    dest: Path,
    *,
    referer: str | None = None,
    timeout_sec: float = 120.0,
) -> tuple[list[Path], list[str]]:
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    warnings: list[str] = []
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if referer:
        headers["Referer"] = referer

    with httpx.Client(follow_redirects=True, timeout=timeout_sec, headers=headers) as client:
        for url, title in links:
            try:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    name = filename_from_response(url, response.headers, title)
                    path = _unique_path(dest, name)
                    with path.open("wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            handle.write(chunk)
                    saved.append(path)
                    logger.info("Downloaded %s -> %s", url, path.name)
            except Exception as exc:
                warnings.append(f"Не удалось скачать {title or url}: {exc}")

    return saved, warnings
