"""Утилиты: распаковка архивов перед чтением документов LlamaIndex."""

from __future__ import annotations

import tempfile
from pathlib import Path

ARCHIVES = {".zip", ".rar"}
MAX_ARCHIVE_DEPTH = 3


def _is_within(directory: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_member(dest: Path, member_name: str) -> Path | None:
    cleaned = member_name.replace("\\", "/").lstrip("/")
    if not cleaned or cleaned.endswith("/"):
        return None
    target = (dest / cleaned).resolve()
    if not _is_within(dest, target):
        raise ValueError(f"Небезопасный путь в архиве: {member_name}")
    return target


def unpack_zip(archive: Path, dest: Path) -> None:
    import zipfile

    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            target = _safe_extract_member(dest, info.filename)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, target.open("wb") as output:
                output.write(source.read())


def unpack_rar(archive: Path, dest: Path) -> None:
    try:
        import rarfile
    except ImportError as exc:
        raise RuntimeError("Для RAR установите пакет rarfile") from exc

    with rarfile.RarFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            target = _safe_extract_member(dest, info.filename)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, target.open("wb") as output:
                output.write(source.read())


def unpack_archive(archive: Path, dest: Path) -> None:
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        unpack_zip(archive, dest)
    elif suffix == ".rar":
        unpack_rar(archive, dest)
    else:
        raise ValueError(f"Неподдерживаемый архив: {archive}")


def expand_archives(
    files: list[tuple[Path, str]],
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
    warnings: list[str],
    depth: int = 0,
) -> list[tuple[Path, str]]:
    """Возвращает (путь_на_диске, отображаемое_имя) для файлов и содержимого архивов."""
    items: list[tuple[Path, str]] = []
    for path, label in files:
        suffix = path.suffix.lower()
        if suffix not in ARCHIVES:
            items.append((path, label))
            continue
        if depth >= MAX_ARCHIVE_DEPTH:
            warnings.append(f"Превышена глубина вложенности архивов: {label}")
            continue
        temp = tempfile.TemporaryDirectory(prefix="ai-tender-archive-")
        temp_dirs.append(temp)
        dest = Path(temp.name)
        try:
            unpack_archive(path, dest)
        except Exception as exc:
            warnings.append(f"Не удалось распаковать {label}: {exc}")
            continue
        nested = [
            (child, f"{label}/{child.relative_to(dest).as_posix()}")
            for child in dest.rglob("*")
            if child.is_file()
        ]
        items.extend(expand_archives(nested, temp_dirs, warnings, depth=depth + 1))
    return items
