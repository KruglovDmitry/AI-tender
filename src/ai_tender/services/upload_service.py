"""Сохранение загрузок из браузера и обновление каталога эталонов."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Protocol

from .archive_service import ARCHIVES, unpack_archive

UPLOADS_ROOT = Path("data/uploads")
ASSETS_ALLOWED_SUFFIXES = {".pdf"}


class UploadedLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name.strip()
    if not base or base in {".", ".."}:
        raise ValueError(f"Некорректное имя файла: {name!r}")
    return base


def new_run_dir(kind: str, root: Path = UPLOADS_ROOT) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / f"{stamp}_{kind}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_files(files: list[UploadedLike], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for item in files:
        name = safe_filename(item.name)
        target = dest / name
        target.write_bytes(item.getvalue())
        saved.append(target)
    return saved


def expand_top_level_archives(folder: Path, warnings: list[str]) -> None:
    """Распаковывает архивы в корне folder и удаляет исходники после успеха."""
    archives = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in ARCHIVES
    )
    for archive in archives:
        unpack_dest = folder / archive.stem
        if unpack_dest.exists():
            unpack_dest = folder / f"{archive.stem}_unpacked"
        if unpack_dest.exists():
            shutil.rmtree(unpack_dest)
        unpack_dest.mkdir(parents=True, exist_ok=True)
        try:
            unpack_archive(archive, unpack_dest)
        except Exception as exc:
            shutil.rmtree(unpack_dest, ignore_errors=True)
            warnings.append(f"Не удалось распаковать {archive.name}: {exc}")
            continue
        # Одна корневая папка с тем же именем, что и архив (2.rar → 2/…) —
        # поднимаем содержимое, чтобы пути не были stem/stem/...
        children = list(unpack_dest.iterdir())
        if (
            len(children) == 1
            and children[0].is_dir()
            and children[0].name == archive.stem
        ):
            nested = children[0]
            for child in nested.iterdir():
                target = unpack_dest / child.name
                if target.exists():
                    target = unpack_dest / f"{child.name}_from_archive"
                shutil.move(str(child), str(target))
            nested.rmdir()
        archive.unlink(missing_ok=True)


def _keep_only_allowed_suffixes(
    folder: Path,
    allowed: set[str],
    warnings: list[str],
) -> None:
    """Удаляет из folder файлы с недопустимым расширением."""
    for path in sorted(folder.rglob("*"), reverse=True):
        if not path.is_file():
            continue
        if path.suffix.lower() in allowed:
            continue
        rel = path.relative_to(folder).as_posix()
        path.unlink(missing_ok=True)
        warnings.append(
            f"Пропущен файл эталона (допускаются только PDF): {rel}"
        )


def clear_directory(folder: Path) -> None:
    """Очищает содержимое folder, не удаляя сам каталог (безопасно для volume mount)."""
    folder.mkdir(parents=True, exist_ok=True)
    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def prepare_upload_dir(files: list[UploadedLike], dest: Path) -> tuple[Path, list[str]]:
    if not files:
        raise ValueError("Не выбраны файлы для загрузки.")
    warnings: list[str] = []
    dest = dest.expanduser().resolve()
    clear_directory(dest)
    save_uploaded_files(files, dest)
    expand_top_level_archives(dest, warnings)
    if not any(path.is_file() for path in dest.rglob("*")):
        raise ValueError("После загрузки не осталось файлов (пустой архив?).")
    return dest, warnings


def replace_shared_assets(
    files: list[UploadedLike],
    assets_root: Path,
) -> tuple[Path, list[str]]:
    """Полностью заменяет содержимое каталога эталонов загруженным пакетом."""
    dest, warnings = prepare_upload_dir(files, assets_root.expanduser().resolve())
    _keep_only_allowed_suffixes(dest, ASSETS_ALLOWED_SUFFIXES, warnings)
    if not any(path.is_file() for path in dest.rglob("*")):
        raise ValueError(
            "После загрузки не осталось PDF-файлов эталонов."
        )
    return dest, warnings


def append_uploaded_files(
    files: list[UploadedLike],
    assets_root: Path,
) -> tuple[Path, list[str], list[str]]:
    """Добавляет файлы в каталог эталонов без очистки существующего содержимого.

    Возвращает (assets_root, warnings, relative_paths новых/изменённых файлов).
    """
    if not files:
        raise ValueError("Не выбраны файлы для загрузки.")
    assets_root = assets_root.expanduser().resolve()
    assets_root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    before = {
        path.relative_to(assets_root).as_posix()
        for path in assets_root.rglob("*")
        if path.is_file()
    }
    before_mtime = {
        path.relative_to(assets_root).as_posix(): path.stat().st_mtime_ns
        for path in assets_root.rglob("*")
        if path.is_file()
    }

    save_uploaded_files(files, assets_root)
    expand_top_level_archives(assets_root, warnings)
    _keep_only_allowed_suffixes(assets_root, ASSETS_ALLOWED_SUFFIXES, warnings)

    after_paths = [
        path for path in assets_root.rglob("*") if path.is_file()
    ]
    if not after_paths and not before:
        raise ValueError(
            "После загрузки не осталось PDF-файлов эталонов."
        )

    changed: list[str] = []
    for path in after_paths:
        rel = path.relative_to(assets_root).as_posix()
        if rel not in before:
            changed.append(rel)
            continue
        try:
            if path.stat().st_mtime_ns != before_mtime.get(rel):
                changed.append(rel)
        except OSError:
            changed.append(rel)

    # Если архив заменил одноимённые файлы с тем же mtime (редко) — всё равно
    # отметим имена загруженных top-level объектов.
    if not changed:
        for item in files:
            name = safe_filename(item.name)
            stem = Path(name).stem
            suffix = Path(name).suffix.lower()
            if suffix in ARCHIVES:
                for path in after_paths:
                    rel = path.relative_to(assets_root).as_posix()
                    if rel == name or rel.startswith(f"{stem}/") or rel.startswith(
                        f"{stem}_unpacked/"
                    ):
                        changed.append(rel)
            elif (assets_root / name).is_file():
                changed.append(name)

    return assets_root, warnings, sorted(set(changed))


def list_files_relative(folder: Path, limit: int = 50) -> list[str]:
    paths = sorted(
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file()
    )
    if len(paths) > limit:
        rest = len(paths) - limit
        return paths[:limit] + [f"… и ещё {rest}"]
    return paths


def cleanup_old_uploads(
    root: Path = UPLOADS_ROOT,
    *,
    max_age_hours: float = 48.0,
) -> int:
    """Удаляет устаревшие каталоги загрузок. Возвращает число удалённых."""
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    return removed
