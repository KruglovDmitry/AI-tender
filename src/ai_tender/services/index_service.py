"""LlamaIndex индекс эталонов (стабильный кэш + инкрементальные add/remove)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from .loader_service import load_documents, split_documents

CURRENT_CACHE_NAME = "current"

_INDEXABLE_SUFFIXES = {".pdf"}


def _resolve_device(device: str | None) -> str | None:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return None


def configure_embeddings(model_name: str, device: str | None = None) -> HuggingFaceEmbedding:
    # batch 8 на CPU очень медленный для ~1000 чанков; 32 заметно быстрее.
    kwargs: dict = {"model_name": model_name, "embed_batch_size": 32}
    resolved = _resolve_device(device)
    if resolved:
        kwargs["device"] = resolved
    embed_model = HuggingFaceEmbedding(**kwargs)
    Settings.embed_model = embed_model
    return embed_model


def folder_fingerprint(folder: Path) -> str:
    """Отпечаток состава папки: относительный путь + размер (без mtime).

    Без mtime, чтобы кэш переживал копирование/распаковку архива на другой машине.
    Смена содержимого при том же размере не инвалидирует кэш (редко для PDF/DOCX).
    """
    parts: list[str] = []
    for path in sorted(_iter_asset_files(folder)):
        relative = path.relative_to(folder).as_posix()
        parts.append(f"{relative}:{path.stat().st_size}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def cache_key(
    folder: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> str:
    """Ключ кэша без абсолютного пути — переносим вместе с проектом к заказчику."""
    raw = (
        f"{folder_fingerprint(folder)}|{embedding_model}|"
        f"{chunk_size}|{chunk_overlap}|ocr={int(ocr_enabled)}|{ocr_languages}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _iter_asset_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in _INDEXABLE_SUFFIXES
    )


def file_fingerprint(path: Path, *, relative: str | None = None) -> dict[str, Any]:
    """Per-file отпечаток: rel_path + size + короткий sha256 содержимого."""
    data = path.read_bytes()
    return {
        "rel_path": relative if relative is not None else path.name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
    }


def scan_assets_files(folder: Path) -> dict[str, dict[str, Any]]:
    """Словарь rel_path → fingerprint для всех индексруемых файлов на диске."""
    result: dict[str, dict[str, Any]] = {}
    for path in _iter_asset_files(folder):
        rel = path.relative_to(folder).as_posix()
        result[rel] = file_fingerprint(path, relative=rel)
    return result


def assets_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / "llama_assets" / CURRENT_CACHE_NAME


def _meta_path(entry_dir: Path) -> Path:
    return entry_dir / "meta.json"


def _read_meta(entry_dir: Path) -> dict[str, Any] | None:
    path = _meta_path(entry_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _params_match(
    meta: dict[str, Any],
    *,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    ocr_enabled: bool,
    ocr_languages: str,
) -> bool:
    return (
        meta.get("embedding_model") == embedding_model
        and meta.get("chunk_size") == chunk_size
        and meta.get("chunk_overlap") == chunk_overlap
        and meta.get("ocr_enabled") == ocr_enabled
        and meta.get("ocr_languages") == ocr_languages
    )


def _stabilize_doc_ids(documents: list[Document]) -> None:
    """Стабильный doc_id = relative file_path для delete_ref_doc / insert."""
    for doc in documents:
        label = str(doc.metadata.get("file_path") or doc.metadata.get("file_name") or "").strip()
        if label:
            doc.doc_id = label


def _files_meta_from_nodes(
    nodes: list[BaseNode],
    disk_fps: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for node in nodes:
        meta = node.metadata or {}
        path = str(meta.get("file_path") or meta.get("file_name") or "")
        if not path or path == "None":
            continue
        counts[path] = counts.get(path, 0) + 1
    files: list[dict[str, Any]] = []
    for path in sorted(counts):
        entry: dict[str, Any] = {
            "rel_path": path,
            "node_count": counts[path],
        }
        if disk_fps and path in disk_fps:
            entry["size"] = disk_fps[path]["size"]
            entry["sha256"] = disk_fps[path]["sha256"]
        files.append(entry)
    return files


def _write_meta(
    entry_dir: Path,
    *,
    assets_path: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    ocr_enabled: bool,
    ocr_languages: str,
    nodes: list[BaseNode],
    warnings: list[str],
) -> dict[str, Any]:
    disk_fps = scan_assets_files(assets_path)
    files = _files_meta_from_nodes(nodes, disk_fps)
    # Сохраняем отпечатки и для файлов, которые есть на диске, но не дали чанков
    # (чтобы sync не пытался бесконечно их добавлять без rebuild).
    indexed_paths = {item["rel_path"] for item in files}
    for rel, fp in disk_fps.items():
        if rel not in indexed_paths:
            files.append(
                {
                    "rel_path": rel,
                    "node_count": 0,
                    "size": fp["size"],
                    "sha256": fp["sha256"],
                }
            )
    files.sort(key=lambda item: item["rel_path"])
    meta = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "ocr_enabled": ocr_enabled,
        "ocr_languages": ocr_languages,
        "folder_fingerprint": folder_fingerprint(assets_path),
        "assets_path": str(assets_path.resolve()),
        "node_count": len(nodes),
        "files": files,
        "indexed_files": [item["rel_path"] for item in files if item.get("node_count", 0) > 0],
        "warnings": warnings,
        "schema_version": 2,
    }
    entry_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(entry_dir).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def _persist_index(index: VectorStoreIndex, entry_dir: Path) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(entry_dir))


def _nodes_from_index(index: VectorStoreIndex) -> list[BaseNode]:
    return list(index.docstore.docs.values())


def _ref_doc_ids_are_stable(index: VectorStoreIndex) -> bool:
    for node in _nodes_from_index(index):
        meta = node.metadata or {}
        path = str(meta.get("file_path") or "")
        if path and node.ref_doc_id != path:
            return False
    return True


def _meta_file_map(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = meta.get("files")
    if isinstance(files, list) and files:
        return {
            str(item["rel_path"]): item
            for item in files
            if isinstance(item, dict) and item.get("rel_path")
        }
    # legacy meta: только список имён
    indexed = meta.get("indexed_files") or []
    return {str(path): {"rel_path": str(path)} for path in indexed}


def _diff_disk_and_meta(
    disk_fps: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Возвращает (to_add, to_remove, to_refresh) по сравнению диска и meta."""
    meta_map = _meta_file_map(meta)
    disk_paths = set(disk_fps)
    meta_paths = set(meta_map)
    to_add = sorted(disk_paths - meta_paths)
    to_remove = sorted(meta_paths - disk_paths)
    to_refresh: list[str] = []
    for path in sorted(disk_paths & meta_paths):
        entry = meta_map[path]
        if "sha256" not in entry or "size" not in entry:
            continue
        fp = disk_fps[path]
        if entry.get("size") != fp["size"] or entry.get("sha256") != fp["sha256"]:
            to_refresh.append(path)
    return to_add, to_remove, to_refresh


def get_assets_index_status(
    assets_path: Path,
    cache_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> dict[str, Any]:
    """Состояние индекса для UI: список файлов, sync, нужна ли пересборка."""
    entry_dir = assets_cache_dir(cache_dir)
    meta = _read_meta(entry_dir)
    disk_fps = scan_assets_files(assets_path)
    ready = bool(meta and (entry_dir / "docstore.json").exists())
    params_ok = bool(meta and _params_match(
        meta,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    ))
    to_add: list[str] = []
    to_remove: list[str] = []
    to_refresh: list[str] = []
    if meta and params_ok:
        to_add, to_remove, to_refresh = _diff_disk_and_meta(disk_fps, meta)
    files = []
    if meta:
        files = list(_meta_file_map(meta).values())
        files.sort(key=lambda item: str(item.get("rel_path", "")))
    return {
        "ready": ready and params_ok,
        "exists": ready,
        "needs_rebuild": ready and not params_ok,
        "out_of_sync": bool(to_add or to_remove or to_refresh),
        "to_add": to_add,
        "to_remove": to_remove,
        "to_refresh": to_refresh,
        "files": files,
        "node_count": int(meta.get("node_count", 0)) if meta else 0,
        "warnings": list(meta.get("warnings", [])) if meta else [],
        "embedding_model": meta.get("embedding_model") if meta else None,
        "chunk_size": meta.get("chunk_size") if meta else None,
        "chunk_overlap": meta.get("chunk_overlap") if meta else None,
        "disk_file_count": len(disk_fps),
    }


def build_index_from_folder(
    folder: Path,
    corpus: str,
    chunk_size: int,
    chunk_overlap: int,
    technical_only: bool = False,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
    only_labels: set[str] | None = None,
) -> tuple[VectorStoreIndex, list[BaseNode], list[str]]:
    documents, warnings = load_documents(
        folder,
        corpus=corpus,
        technical_only=technical_only,
        only_labels=only_labels,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    if not documents:
        details = "; ".join(warnings) if warnings else "файлы не найдены или пусты"
        raise ValueError(f"Не удалось извлечь документы из {folder}. {details}")
    _stabilize_doc_ids(documents)
    nodes = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not nodes:
        raise ValueError(f"После разбиения на чанки документов нет: {folder}")
    index = VectorStoreIndex(nodes, show_progress=True)
    return index, nodes, warnings


def _load_persisted_index(entry_dir: Path) -> VectorStoreIndex:
    storage = StorageContext.from_defaults(persist_dir=str(entry_dir))
    return load_index_from_storage(storage)


def _try_migrate_legacy_cache(
    assets_path: Path,
    cache_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    ocr_enabled: bool,
    ocr_languages: str,
) -> bool:
    """Копирует старый hash-кэш в current/, если параметры и fingerprint совпадают."""
    entry_dir = assets_cache_dir(cache_dir)
    if (entry_dir / "docstore.json").exists():
        return False
    key = cache_key(
        assets_path,
        embedding_model,
        chunk_size,
        chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    legacy = cache_dir / "llama_assets" / key
    if not (legacy / "docstore.json").exists():
        return False
    legacy_meta = _read_meta(legacy)
    if not legacy_meta or not _params_match(
        legacy_meta,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    ):
        return False
    if legacy_meta.get("folder_fingerprint") != folder_fingerprint(assets_path):
        return False
    if entry_dir.exists():
        shutil.rmtree(entry_dir)
    shutil.copytree(legacy, entry_dir)
    # Обновим meta до schema v2 с per-file отпечатками
    try:
        index = _load_persisted_index(entry_dir)
        nodes = _nodes_from_index(index)
        _write_meta(
            entry_dir,
            assets_path=assets_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
            nodes=nodes,
            warnings=list(legacy_meta.get("warnings", [])),
        )
    except Exception:
        shutil.rmtree(entry_dir, ignore_errors=True)
        return False
    return True


def rebuild_assets_index(
    assets_path: Path,
    cache_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    device: str | None = None,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[VectorStoreIndex, list[BaseNode], list[str]]:
    """Полная пересборка индекса в data/cache/llama_assets/current/."""
    configure_embeddings(embedding_model, device)
    assets_path = assets_path.expanduser().resolve()
    entry_dir = assets_cache_dir(cache_dir)
    if entry_dir.exists():
        shutil.rmtree(entry_dir)

    index, nodes, warnings = build_index_from_folder(
        assets_path,
        corpus="assets",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    _persist_index(index, entry_dir)
    _write_meta(
        entry_dir,
        assets_path=assets_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
        nodes=nodes,
        warnings=warnings,
    )
    return index, nodes, warnings


def _delete_paths_from_index(index: VectorStoreIndex, paths: list[str]) -> None:
    for path in paths:
        try:
            index.delete_ref_doc(path, delete_from_docstore=True)
        except Exception:
            # Fallback: удалить узлы с этим file_path (старый кэш без стабильного doc_id).
            to_drop = [
                node_id
                for node_id, node in list(index.docstore.docs.items())
                if (node.metadata or {}).get("file_path") == path
                or (node.metadata or {}).get("file_name") == path
            ]
            for node_id in to_drop:
                index.delete_nodes([node_id], delete_from_docstore=True)


def _load_nodes_for_labels(
    assets_path: Path,
    labels: set[str],
    chunk_size: int,
    chunk_overlap: int,
    *,
    ocr_enabled: bool,
    ocr_languages: str,
) -> tuple[list[BaseNode], list[str]]:
    if not labels:
        return [], []
    documents, warnings = load_documents(
        assets_path,
        corpus="assets",
        only_labels=labels,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    if not documents:
        return [], warnings
    _stabilize_doc_ids(documents)
    nodes = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return nodes, warnings


def add_assets_to_index(
    assets_path: Path,
    cache_dir: Path,
    relative_paths: list[str],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    device: str | None = None,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[VectorStoreIndex, list[BaseNode], list[str]]:
    """Добавляет/обновляет файлы в индексе (только новые эмбеддинги)."""
    configure_embeddings(embedding_model, device)
    assets_path = assets_path.expanduser().resolve()
    entry_dir = assets_cache_dir(cache_dir)
    labels = sorted({path.replace("\\", "/").lstrip("/") for path in relative_paths if path})
    if not labels:
        raise ValueError("Не указаны файлы для добавления в индекс.")

    meta = _read_meta(entry_dir)
    if not meta or not (entry_dir / "docstore.json").exists():
        return rebuild_assets_index(
            assets_path,
            cache_dir,
            embedding_model,
            chunk_size,
            chunk_overlap,
            device=device,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        )
    if not _params_match(
        meta,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    ):
        raise ValueError(
            "Параметры embedding/chunk/OCR изменились — нужна полная пересборка индекса."
        )

    index = _load_persisted_index(entry_dir)
    if not _ref_doc_ids_are_stable(index):
        return rebuild_assets_index(
            assets_path,
            cache_dir,
            embedding_model,
            chunk_size,
            chunk_overlap,
            device=device,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        )

    _delete_paths_from_index(index, labels)
    nodes_new, warnings = _load_nodes_for_labels(
        assets_path,
        set(labels),
        chunk_size,
        chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    if nodes_new:
        index.insert_nodes(nodes_new)

    all_nodes = _nodes_from_index(index)
    _persist_index(index, entry_dir)
    _write_meta(
        entry_dir,
        assets_path=assets_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
        nodes=all_nodes,
        warnings=list(meta.get("warnings", [])) + warnings,
    )
    return index, all_nodes, warnings


def remove_asset_from_index(
    assets_path: Path,
    cache_dir: Path,
    relative_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    device: str | None = None,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
    *,
    delete_file: bool = True,
) -> tuple[VectorStoreIndex | None, list[BaseNode], list[str]]:
    """Удаляет файл из индекса и (по умолчанию) с диска."""
    configure_embeddings(embedding_model, device)
    assets_path = assets_path.expanduser().resolve()
    entry_dir = assets_cache_dir(cache_dir)
    rel = relative_path.replace("\\", "/").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise ValueError(f"Некорректный путь эталона: {relative_path!r}")

    meta = _read_meta(entry_dir)
    warnings: list[str] = []
    if not meta or not (entry_dir / "docstore.json").exists():
        if delete_file:
            target = (assets_path / rel).resolve()
            if assets_path.resolve() not in target.parents and target != assets_path.resolve():
                raise ValueError(f"Путь вне каталога эталонов: {rel}")
            if target.is_file():
                target.unlink()
        return None, [], warnings

    if not _params_match(
        meta,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    ):
        raise ValueError(
            "Параметры embedding/chunk/OCR изменились — нужна полная пересборка индекса."
        )

    index = _load_persisted_index(entry_dir)
    if not _ref_doc_ids_are_stable(index):
        # Без стабильных id надёжнее пересобрать после удаления файла.
        if delete_file:
            target = assets_path / rel
            if target.is_file():
                target.unlink()
        return rebuild_assets_index(
            assets_path,
            cache_dir,
            embedding_model,
            chunk_size,
            chunk_overlap,
            device=device,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        )

    _delete_paths_from_index(index, [rel])
    from .indexing.persistance import delete_product_artifacts

    delete_product_artifacts(cache_dir, rel)
    if delete_file:
        target = (assets_path / rel).resolve()
        if not str(target).startswith(str(assets_path.resolve())):
            raise ValueError(f"Путь вне каталога эталонов: {rel}")
        if target.is_file():
            target.unlink()
        else:
            warnings.append(f"Файл на диске не найден: {rel}")

    all_nodes = _nodes_from_index(index)
    if not all_nodes:
        shutil.rmtree(entry_dir, ignore_errors=True)
        if assets_path.is_dir() and any(_iter_asset_files(assets_path)):
            warnings.append("Индекс пуст после удаления; на диске остались файлы.")
        return None, [], warnings

    _persist_index(index, entry_dir)
    _write_meta(
        entry_dir,
        assets_path=assets_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
        nodes=all_nodes,
        warnings=list(meta.get("warnings", [])) + warnings,
    )
    return index, all_nodes, warnings


def _sync_index_with_disk(
    index: VectorStoreIndex,
    assets_path: Path,
    entry_dir: Path,
    meta: dict[str, Any],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    ocr_enabled: bool,
    ocr_languages: str,
) -> tuple[VectorStoreIndex, list[BaseNode], list[str], bool]:
    """Инкрементально догоняет индекс до состояния assets/ на диске."""
    disk_fps = scan_assets_files(assets_path)
    to_add, to_remove, to_refresh = _diff_disk_and_meta(disk_fps, meta)
    if not (to_add or to_remove or to_refresh):
        nodes = _nodes_from_index(index)
        return index, nodes, list(meta.get("warnings", [])), True

    warnings = list(meta.get("warnings", []))
    changed = False

    if to_remove:
        _delete_paths_from_index(index, to_remove)
        changed = True

    refresh_and_add = sorted(set(to_add) | set(to_refresh))
    if refresh_and_add:
        _delete_paths_from_index(index, refresh_and_add)
        nodes_new, load_warnings = _load_nodes_for_labels(
            assets_path,
            set(refresh_and_add),
            chunk_size,
            chunk_overlap,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        )
        warnings.extend(load_warnings)
        if nodes_new:
            index.insert_nodes(nodes_new)
        changed = True

    all_nodes = _nodes_from_index(index)
    if changed:
        _persist_index(index, entry_dir)
        _write_meta(
            entry_dir,
            assets_path=assets_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
            nodes=all_nodes,
            warnings=warnings,
        )
    return index, all_nodes, warnings, False


def load_or_build_assets_index(
    assets_path: Path,
    cache_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    device: str | None = None,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[VectorStoreIndex, list[BaseNode], list[str], bool]:
    configure_embeddings(embedding_model, device)
    assets_path = assets_path.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    entry_dir = assets_cache_dir(cache_dir)

    _try_migrate_legacy_cache(
        assets_path,
        cache_dir,
        embedding_model,
        chunk_size,
        chunk_overlap,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )

    meta = _read_meta(entry_dir)
    if meta and (entry_dir / "docstore.json").exists():
        if not _params_match(
            meta,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        ):
            index, nodes, warnings = rebuild_assets_index(
                assets_path,
                cache_dir,
                embedding_model,
                chunk_size,
                chunk_overlap,
                device=device,
                ocr_enabled=ocr_enabled,
                ocr_languages=ocr_languages,
            )
            return index, nodes, warnings, False

        index = _load_persisted_index(entry_dir)
        disk_fps = scan_assets_files(assets_path)
        to_add, to_remove, to_refresh = _diff_disk_and_meta(disk_fps, meta)
        if not (to_add or to_remove or to_refresh):
            nodes = _nodes_from_index(index)
            return index, nodes, list(meta.get("warnings", [])), True

        if not _ref_doc_ids_are_stable(index):
            index, nodes, warnings = rebuild_assets_index(
                assets_path,
                cache_dir,
                embedding_model,
                chunk_size,
                chunk_overlap,
                device=device,
                ocr_enabled=ocr_enabled,
                ocr_languages=ocr_languages,
            )
            return index, nodes, warnings, False

        index, nodes, warnings, reused = _sync_index_with_disk(
            index,
            assets_path,
            entry_dir,
            meta,
            embedding_model,
            chunk_size,
            chunk_overlap,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        )
        return index, nodes, warnings, reused

    index, nodes, warnings = rebuild_assets_index(
        assets_path,
        cache_dir,
        embedding_model,
        chunk_size,
        chunk_overlap,
        device=device,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    return index, nodes, warnings, False


def _rrf_fuse(
    result_lists: list[list[NodeWithScore]],
    top_k: int,
    k: int = 60,
) -> list[NodeWithScore]:
    """Reciprocal Rank Fusion без LLM."""
    scores: dict[str, float] = {}
    nodes: dict[str, NodeWithScore] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            node_id = hit.node.node_id
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)
            current = nodes.get(node_id)
            if current is None or (hit.score or 0) > (current.score or 0):
                nodes[node_id] = hit
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    output: list[NodeWithScore] = []
    for node_id, rrf_score in fused[:top_k]:
        hit = nodes[node_id]
        output.append(NodeWithScore(node=hit.node, score=rrf_score))
    return output


def build_bm25_retriever(index: VectorStoreIndex, top_k: int) -> BM25Retriever:
    try:
        return BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=top_k,
            language="russian",
        )
    except Exception:
        return BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=top_k,
        )


def _page_from_metadata(meta: dict) -> int | None:
    raw = meta.get("page_number")
    if raw is None:
        raw = meta.get("page_label")
    if raw is None:
        location = str(meta.get("location") or "")
        if location.startswith("стр."):
            raw = location.removeprefix("стр.").strip()
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def node_to_evidence(node: BaseNode, score: float | None = None):
    from ..models import Evidence

    meta = node.metadata or {}
    text = node.get_content(metadata_mode="none")
    quote = " ".join(text.split())
    if len(quote) > 1600:
        quote = quote[:1599] + "…"
    return Evidence(
        file=str(meta.get("file_path") or meta.get("file_name") or "unknown"),
        location=str(meta.get("location") or "фрагмент"),
        quote=quote,
        score=None if score is None else round(float(score), 4),
        page=_page_from_metadata(meta),
        line_start=_optional_int(meta.get("line_start")),
        line_end=_optional_int(meta.get("line_end")),
    )


def _optional_int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def indexed_file_paths(nodes: list[BaseNode]) -> list[str]:
    files = {
        str(node.metadata.get("file_path") or node.metadata.get("file_name"))
        for node in nodes
        if node.metadata
    }
    return sorted(path for path in files if path and path != "None")


def retrieve_for_queries(
    search_index: VectorStoreIndex,
    queries: list[str],
    top_k: int,
) -> list[list[NodeWithScore]]:
    """Hybrid retrieval по списку текстовых запросов (извлечённые требования)."""
    bm25 = build_bm25_retriever(search_index, top_k)
    vector_retriever = search_index.as_retriever(similarity_top_k=top_k)
    output: list[list[NodeWithScore]] = []
    for raw in queries:
        query = " ".join((raw or "").split())[:800]
        if not query:
            output.append([])
            continue
        vector_hits = vector_retriever.retrieve(query)
        bm25_hits = bm25.retrieve(query)
        output.append(_rrf_fuse([vector_hits, bm25_hits], top_k=top_k)[:top_k])
    return output
