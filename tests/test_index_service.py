from pathlib import Path
import json
import os

from docx import Document
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding

from ai_tender.services.index_service import (
    add_assets_to_index,
    assets_cache_dir,
    cache_key,
    file_fingerprint,
    folder_fingerprint,
    get_assets_index_status,
    indexed_file_paths,
    load_or_build_assets_index,
    rebuild_assets_index,
    remove_asset_from_index,
    scan_assets_files,
)


def _write_docx(path: Path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(path)


def _long_text(label: str) -> str:
    return (
        f"Эталонный документ {label}. "
        "Требования по степени защиты IP54, номинальному току и климатическому исполнению. "
    ) * 8


def _use_mock_embeddings(monkeypatch) -> None:
    embed = MockEmbedding(embed_dim=8)

    def _configure(model_name: str, device: str | None = None):
        Settings.embed_model = embed
        return embed

    monkeypatch.setattr(
        "ai_tender.services.index_service.configure_embeddings",
        _configure,
    )
    Settings.embed_model = embed


def test_folder_fingerprint_changes_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "эталон.docx"
    _write_docx(path, "Исходный текст эталона с параметром IP54.")
    first = folder_fingerprint(tmp_path)
    _write_docx(path, "Обновлённый текст эталона с параметром IP54 и током.")
    second = folder_fingerprint(tmp_path)
    assert first != second


def test_folder_fingerprint_ignores_mtime(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("эталон", encoding="utf-8")
    first = folder_fingerprint(tmp_path)
    os.utime(path, (1_700_000_000, 1_800_000_000))
    second = folder_fingerprint(tmp_path)
    assert first == second


def test_cache_key_includes_chunk_settings(tmp_path: Path) -> None:
    _write_docx(tmp_path / "a.docx", "Достаточно длинный текст эталонного документа.")
    key_a = cache_key(tmp_path, "BAAI/bge-m3", 1024, 128)
    key_b = cache_key(tmp_path, "BAAI/bge-m3", 512, 128)
    assert key_a != key_b


def test_cache_key_includes_ocr_settings(tmp_path: Path) -> None:
    _write_docx(tmp_path / "a.docx", "Достаточно длинный текст эталонного документа.")
    key_on = cache_key(tmp_path, "BAAI/bge-m3", 1024, 128, ocr_enabled=True)
    key_off = cache_key(tmp_path, "BAAI/bge-m3", 1024, 128, ocr_enabled=False)
    key_lang = cache_key(
        tmp_path, "BAAI/bge-m3", 1024, 128, ocr_enabled=True, ocr_languages="eng"
    )
    assert key_on != key_off
    assert key_on != key_lang


def test_cache_key_portable_across_folder_paths(tmp_path: Path) -> None:
    left = tmp_path / "machine_a" / "assets"
    right = tmp_path / "machine_b" / "assets"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    (left / "spec.txt").write_text("одна и та же спецификация", encoding="utf-8")
    (right / "spec.txt").write_text("одна и та же спецификация", encoding="utf-8")
    assert cache_key(left, "BAAI/bge-m3", 1024, 128) == cache_key(
        right, "BAAI/bge-m3", 1024, 128
    )


def test_file_fingerprint_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("alpha", encoding="utf-8")
    first = file_fingerprint(path, relative="a.txt")
    path.write_text("beta!", encoding="utf-8")
    second = file_fingerprint(path, relative="a.txt")
    assert first["sha256"] != second["sha256"]
    assert first["rel_path"] == "a.txt"


def test_rebuild_and_reuse_current_cache(tmp_path: Path, monkeypatch) -> None:
    _use_mock_embeddings(monkeypatch)
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "one.txt").write_text(_long_text("one"), encoding="utf-8")

    index, nodes, warnings = rebuild_assets_index(
        assets, cache, "mock", 256, 32, ocr_enabled=False
    )
    assert not warnings or isinstance(warnings, list)
    assert indexed_file_paths(nodes) == ["one.txt"]
    entry = assets_cache_dir(cache)
    assert (entry / "meta.json").exists()
    meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 2
    assert meta["files"][0]["rel_path"] == "one.txt"
    assert "sha256" in meta["files"][0]

    index2, nodes2, warnings2, reused = load_or_build_assets_index(
        assets, cache, "mock", 256, 32, ocr_enabled=False
    )
    assert reused is True
    assert indexed_file_paths(nodes2) == ["one.txt"]
    assert index2 is not None


def test_add_and_remove_assets_incremental(tmp_path: Path, monkeypatch) -> None:
    _use_mock_embeddings(monkeypatch)
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "a.txt").write_text(_long_text("a"), encoding="utf-8")

    rebuild_assets_index(assets, cache, "mock", 256, 32, ocr_enabled=False)

    (assets / "b.txt").write_text(_long_text("b"), encoding="utf-8")
    _index, nodes, _warnings = add_assets_to_index(
        assets,
        cache,
        ["b.txt"],
        "mock",
        256,
        32,
        ocr_enabled=False,
    )
    assert indexed_file_paths(nodes) == ["a.txt", "b.txt"]

    status = get_assets_index_status(
        assets, cache, "mock", 256, 32, ocr_enabled=False
    )
    assert status["ready"] is True
    assert status["out_of_sync"] is False
    assert {f["rel_path"] for f in status["files"]} >= {"a.txt", "b.txt"}

    _index2, nodes2, _w2 = remove_asset_from_index(
        assets,
        cache,
        "a.txt",
        "mock",
        256,
        32,
        ocr_enabled=False,
    )
    assert indexed_file_paths(nodes2) == ["b.txt"]
    assert not (assets / "a.txt").exists()
    assert (assets / "b.txt").exists()


def test_load_syncs_new_disk_file(tmp_path: Path, monkeypatch) -> None:
    _use_mock_embeddings(monkeypatch)
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "a.txt").write_text(_long_text("a"), encoding="utf-8")
    rebuild_assets_index(assets, cache, "mock", 256, 32, ocr_enabled=False)

    (assets / "extra.txt").write_text(_long_text("extra"), encoding="utf-8")
    status = get_assets_index_status(
        assets, cache, "mock", 256, 32, ocr_enabled=False
    )
    assert status["out_of_sync"] is True
    assert "extra.txt" in status["to_add"]

    _index, nodes, _warnings, reused = load_or_build_assets_index(
        assets, cache, "mock", 256, 32, ocr_enabled=False
    )
    assert reused is False
    assert "extra.txt" in indexed_file_paths(nodes)


def test_scan_assets_files(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    scanned = scan_assets_files(tmp_path)
    assert list(scanned) == ["keep.txt"]
