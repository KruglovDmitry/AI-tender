from pathlib import Path
from unittest.mock import MagicMock

from ai_tender.nodes.select_files import (
    TenderFileEntry,
    build_catalog_from_inventory,
    format_catalog_tree,
    ranked_file_paths,
    select_files_heuristic,
    select_tender_files_by_llm,
)
from ai_tender.loaders import inventory_tender_folder


def _entry(path: str, suffix: str = ".pdf", size: int = 1024) -> object:
    parent = Path(path).parent.as_posix()
    if parent == ".":
        parent = ""
    return TenderFileEntry(path=path, suffix=suffix, size_bytes=size, parent=parent)


def test_heuristic_prefers_detailed_tz() -> None:
    entries = [
        _entry("Приложение №2 Проект договора.docx", ".docx"),
        _entry("Приложение №1 Техническое задание/ТЗ на провед.закупки.pdf"),
        _entry("Приложение №1 Техническое задание/ТЗ ФЗ-522_6-20кВ.pdf"),
        _entry("Извещение по конкурсу.docx", ".docx"),
        _entry("Приложения к ТЗ/Расчет НМЦ.pdf"),
    ]
    result = select_files_heuristic(entries, max_files=3)
    paths = [item["path"] for item in result["files"]]
    assert "Приложение №1 Техническое задание/ТЗ ФЗ-522_6-20кВ.pdf" in paths
    assert "Приложение №2 Проект договора.docx" not in paths
    by_path = {item["path"]: item for item in result["files"]}
    detailed = by_path["Приложение №1 Техническое задание/ТЗ ФЗ-522_6-20кВ.pdf"]
    assert detailed["priority"] == 1
    assert detailed["scope_level"] == 2


def test_ranked_file_paths_sorts_by_priority() -> None:
    selection = {
        "files": [
            {"path": "b.pdf", "priority": 2, "scope_level": 2},
            {"path": "a.pdf", "priority": 1, "scope_level": 1},
        ]
    }
    assert ranked_file_paths(selection) == ["a.pdf", "b.pdf"]


def test_select_by_llm_parses_response() -> None:
    entries = [
        _entry("ТЗ/основное.pdf"),
        _entry("договор.docx", ".docx"),
    ]
    llm = MagicMock()
    llm.complete.return_value = (
        '{"files": [{"path": "ТЗ/основное.pdf", "priority": 1, "scope_level": 1, '
        '"role": "tz_main", "reason": "главное ТЗ"}], '
        '"skip": [{"path": "договор.docx", "reason": "шаблон"}]}'
    )
    result = select_tender_files_by_llm(entries, "tree", llm, max_files=3)
    assert result["files"][0]["path"] == "ТЗ/основное.pdf"
    assert result["mode"] == "llm"


def test_build_catalog_from_inventory(tmp_path: Path) -> None:
    tz_dir = tmp_path / "Приложение №1 Техническое задание"
    tz_dir.mkdir()
    (tz_dir / "ТЗ.pdf").write_bytes(b"x" * 50)
    (tmp_path / "договор.docx").write_bytes(b"x" * 10)

    inventory = inventory_tender_folder(tmp_path)
    try:
        entries = build_catalog_from_inventory(inventory)
        paths = {entry.path for entry in entries}
        assert "Приложение №1 Техническое задание/ТЗ.pdf" in paths
        tree = format_catalog_tree(entries)
        assert "ТЗ.pdf" in tree
    finally:
        inventory.cleanup()


def test_select_by_llm_rejects_unknown_paths() -> None:
    entries = [_entry("real.pdf")]
    llm = MagicMock()
    llm.complete.return_value = (
        '{"files": [{"path": "missing.pdf", "priority": 1}], "skip": []}'
    )
    result = select_tender_files_by_llm(entries, "tree", llm, max_files=3)
    assert result["files"] == []
