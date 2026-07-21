"""Чтение legacy Microsoft Office (.doc) без MS Word."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

LEGACY_WORD_SUFFIXES = {".doc", ".dot"}


def extract_doc_text(path: Path, *, timeout_sec: int = 90) -> tuple[str, str | None]:
    """
    Извлекает текст из Word 97–2003 (.doc).
    Возвращает (text, error). error=None при успехе.
    """
    try:
        import sharepoint2text

        result = next(sharepoint2text.read_file(path))
        text = (result.get_full_text() or "").strip()
        if text:
            return text, None
        return "", "Пустой текст после sharepoint-to-text"
    except ImportError:
        pass
    except StopIteration:
        return "", "sharepoint-to-text: пустой результат"
    except Exception as exc:
        return "", f"sharepoint-to-text: {exc}"

    antiword = shutil.which("antiword")
    if antiword:
        try:
            proc = subprocess.run(
                [antiword, "-m", "UTF-8.txt", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
            text = (proc.stdout or "").strip()
            if proc.returncode == 0 and text:
                return text, None
            err = (proc.stderr or "").strip() or f"antiword: код {proc.returncode}"
            return text, err if not text else None
        except Exception as exc:
            return "", f"antiword: {exc}"

    return (
        "",
        "Формат .doc: установите пакет sharepoint-to-text (pip) или antiword в PATH",
    )
