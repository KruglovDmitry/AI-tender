"""Распаковка ZIP/RAR перед чтением документов."""

from __future__ import annotations

import os
import shutil
import subprocess
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


def find_unrar_tool() -> Path | None:
    env = os.environ.get("UNRAR_TOOL")
    if env and Path(env).is_file():
        return Path(env)

    home_local = Path.home() / ".local" / "bin" / "unrar"
    for candidate in (
        shutil.which("unrar"),
        shutil.which("unrar-free"),
        shutil.which("UnRAR"),
        str(home_local) if home_local.is_file() else None,
        "/usr/bin/unrar",
        "/usr/bin/unrar-free",
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _nonempty_files(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.stat().st_size > 0
    ]


def _assert_extracted_payload(dest: Path, archive: Path) -> None:
    """7-Zip на RAR5 часто создаёт пустые файлы и всё равно пишет exit≠0/0."""
    if _nonempty_files(dest):
        return
    raise RuntimeError(
        f"После распаковки {archive.name} нет файлов с данными "
        "(часто системный 7z не умеет метод сжатия RAR5 — нужен UnRAR)."
    )


def find_7z_tool() -> Path | None:
    env = os.environ.get("SEVEN_ZIP_CMD") or os.environ.get("7Z_CMD")
    if env and Path(env).is_file():
        return Path(env)

    for candidate in (
        shutil.which("7z"),
        shutil.which("7zz"),
        shutil.which("7za"),
        "/usr/bin/7z",
        "/usr/bin/7zz",
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def configure_rarfile() -> Path | None:
    """Настраивает rarfile на найденный UnRAR; возвращает путь к утилите."""
    tool = find_unrar_tool()
    if tool is None:
        return None
    import rarfile

    rarfile.UNRAR_TOOL = str(tool)
    return tool


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


def _unpack_rar_with_unrar(archive: Path, dest: Path, unrar: Path) -> None:
    """Прямой вызов UnRAR надёжнее rarfile на RAR5."""
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(unrar), "x", "-o+", "-y", str(archive), str(dest) + os.sep],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "неизвестная ошибка").strip()
        raise RuntimeError(details)
    _assert_extracted_payload(dest, archive)


def _unpack_rar_with_7z(archive: Path, dest: Path, seven_zip: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(seven_zip), "x", str(archive), f"-o{dest}", "-y"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{result.stderr or ''}\n{result.stdout or ''}"
    if result.returncode != 0 or "Unsupported Method" in combined:
        details = combined.strip() or "неизвестная ошибка"
        raise RuntimeError(details)
    _assert_extracted_payload(dest, archive)


def unpack_rar(archive: Path, dest: Path) -> None:
    tool = find_unrar_tool()
    if tool is not None:
        _unpack_rar_with_unrar(archive, dest, tool)
        return

    seven_zip = find_7z_tool()
    if seven_zip is not None:
        _unpack_rar_with_7z(archive, dest, seven_zip)
        return

    raise RuntimeError(
        "Не найден UnRAR или 7-Zip. Установите WinRAR/UnRAR/7-Zip или задайте UNRAR_TOOL "
        "(например C:\\Program Files\\WinRAR\\UnRAR.exe)."
    )


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
